//! MoE expert offload with disk backing (PADDOCK_MOE_DISK) on the serial
//! path, against the pinned backing (PADDOCK_MOE_HOST) on the same 35B-A3B
//! UD file. Four gates, each run alone (`--exact`): the expert switches are
//! process-wide, and the disk gate's peak RSS must be its own.
//!
//! 1. `moe_disk_write_artifact` - writes the artifact named by
//!    QWEN36_MOE_DISK_ARTIFACT from the GGUF (kept if it exists), then a
//!    pinned load writes a reference beside it (`<artifact>.ref`: prefill
//!    logits + greedy tokens).
//! 2. `moe_disk_parity_and_measure` - disk load; the same prompt must give
//!    the reference's logits and tokens bit for bit, then one cold and
//!    QWEN36_MOE_DISK_WARM warm runs report TTFT, prefill and decode tok/s,
//!    misses, bytes and read rate, staging waits, device read bytes and the
//!    process peak RSS. Batched serving must refuse.
//! 3. `moe_pinned_measure` - the same runs on the pinned backing, the
//!    like-for-like baseline (pinned replays its graphs, disk runs eager).
//! 4. `moe_disk_refuses_a_foreign_artifact` - a file that is not this
//!    model's artifact fails the load with a sentence naming it.
//!
//! Both pin the token-batched expert class (PADDOCK_NO_SORTED_QMOE=1): the
//! disk path serves every width through it, so that is the class the pinned
//! reference must run for a bit-exact comparison.
mod common;

use std::path::{Path, PathBuf};
use std::time::Instant;

use paddock_engine::gpu_model::qwen35::GpuQwen35;
use paddock_models::mapped::MappedGguf;

const MAX_CTX: usize = 4608;
const PARITY_PROMPT: usize = 600;
const PARITY_TOKENS: usize = 32;

fn setup() -> Option<(
    PathBuf,
    PathBuf,
    std::sync::Arc<paddock_engine::gpu::GpuExecutor>,
)> {
    if !common::heavy() {
        return None;
    }
    let model = common::model("QWEN36_MOE_UD_GGUF", common::QWEN36_35B_A3B_UD_Q4)?;
    let Some(artifact) = std::env::var_os("QWEN36_MOE_DISK_ARTIFACT").map(PathBuf::from) else {
        eprintln!("SKIP: set QWEN36_MOE_DISK_ARTIFACT to the artifact path");
        return None;
    };
    let exec = common::gpu_arc()?;
    // SAFETY: single-threaded test binary (run with --exact), set before any reader.
    unsafe {
        std::env::set_var("PADDOCK_NO_SORTED_QMOE", "1");
        std::env::set_var("PADDOCK_NO_SPEC", "1");
        if std::env::var_os("PADDOCK_MOE_CACHE_SLOTS").is_none() {
            std::env::set_var("PADDOCK_MOE_CACHE_SLOTS", "32");
        }
    }
    Some((model, artifact, exec))
}

/// A deterministic synthetic prompt of `n` ordinary-vocabulary tokens.
fn prompt(n: usize) -> Vec<u32> {
    (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// Prefill logits and `n` greedy tokens, through the serial seam.
fn run(m: &mut GpuQwen35, prompt: &[u32], n: usize) -> (Vec<f32>, Vec<u32>) {
    m.reset();
    let logits = m.prefill(prompt).expect("prefill");
    let mut tok = argmax(&logits);
    let mut out = vec![tok];
    for _ in 1..n {
        tok = argmax(&m.forward_one(tok).expect("decode"));
        out.push(tok);
    }
    (logits, out)
}

fn ref_path(artifact: &Path) -> PathBuf {
    let mut p = artifact.as_os_str().to_owned();
    p.push(".ref");
    p.into()
}

fn write_ref(path: &Path, logits: &[f32], tokens: &[u32]) {
    let mut b = Vec::new();
    b.extend_from_slice(&(logits.len() as u32).to_le_bytes());
    logits
        .iter()
        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    b.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    tokens
        .iter()
        .for_each(|x| b.extend_from_slice(&x.to_le_bytes()));
    std::fs::write(path, b).expect("write reference");
}

fn read_ref(path: &Path) -> (Vec<f32>, Vec<u32>) {
    let b = std::fs::read(path).expect("pinned reference: run moe_disk_write_artifact first");
    let word = |i: usize| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().expect("4 bytes"));
    let nl = word(0) as usize;
    let logits = (0..nl).map(|i| f32::from_bits(word(1 + i))).collect();
    let nt = word(1 + nl) as usize;
    let tokens = (0..nt).map(|i| word(2 + nl + i)).collect();
    (logits, tokens)
}

/// Evict the artifact's clean pages so the first run reads from the device.
fn drop_page_cache(path: &Path) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let f = std::fs::File::open(path).expect("open artifact");
        // SAFETY: plain advisory call on a descriptor we hold.
        let rc = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        eprintln!("page cache dropped for the artifact: rc={rc}");
    }
}

fn proc_field(file: &str, key: &str) -> Option<u64> {
    let s = std::fs::read_to_string(file).ok()?;
    let line = s.lines().find(|l| l.starts_with(key))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// One cold and QWEN36_MOE_DISK_WARM warm runs of a QWEN36_MOE_DISK_PROMPT
/// token prompt plus QWEN36_MOE_DISK_DECODE greedy tokens; the disk counters
/// per phase when the backing is disk.
fn measure(m: &mut GpuQwen35, artifact: Option<&Path>) {
    use paddock_engine::gpu::MoeDiskStats;
    let n_prompt = env_usize("QWEN36_MOE_DISK_PROMPT", 4000);
    let n_decode = env_usize("QWEN36_MOE_DISK_DECODE", 64);
    let warm = env_usize("QWEN36_MOE_DISK_WARM", 5);
    let p = prompt(n_prompt);
    if let Some(a) = artifact {
        drop_page_cache(a);
    }
    let line = |a: Option<MoeDiskStats>, b: Option<MoeDiskStats>| {
        let (Some(a), Some(b)) = (a, b) else {
            return String::from("pinned");
        };
        let rows = b.rows - a.rows;
        let misses = b.misses - a.misses;
        let mib = (b.bytes_read - a.bytes_read) as f64 / (1u64 << 20) as f64;
        let rt = (b.read_time - a.read_time).as_secs_f64();
        format!(
            "misses {misses} of {rows} rows (hit {:.1}%), read {mib:.0} MiB in {rt:.2}s \
             ({:.0} MiB/s), staging waits {}",
            100.0 * (1.0 - misses as f64 / rows.max(1) as f64),
            mib / rt.max(1e-9),
            b.staging_waits - a.staging_waits
        )
    };
    for r in 0..=warm {
        let s0 = m.moe_disk_stats();
        m.reset();
        let t0 = Instant::now();
        let logits = m.prefill(&p).expect("prefill");
        let ttft = t0.elapsed().as_secs_f64();
        let s1 = m.moe_disk_stats();
        let mut tok = argmax(&logits);
        let t1 = Instant::now();
        for _ in 0..n_decode {
            tok = argmax(&m.forward_one(tok).expect("decode"));
        }
        let dec = t1.elapsed().as_secs_f64();
        let s2 = m.moe_disk_stats();
        eprintln!(
            "run {r} {}: ttft {:.0} ms ({:.1} prefill tok/s, {n_prompt} tokens) | {}",
            if r == 0 { "cold" } else { "warm" },
            ttft * 1e3,
            n_prompt as f64 / ttft,
            line(s0, s1)
        );
        eprintln!(
            "run {r} decode: {:.2} tok/s over {n_decode} tokens | {}",
            n_decode as f64 / dec,
            line(s1, s2)
        );
    }
    eprintln!("disk totals: {:?}", m.moe_disk_stats());
    eprintln!(
        "peak RSS (VmHWM): {:?} KiB, device read_bytes: {:?}",
        proc_field("/proc/self/status", "VmHWM:"),
        proc_field("/proc/self/io", "read_bytes:")
    );
}

#[test]
fn moe_disk_write_artifact() {
    let Some((model, artifact, exec)) = setup() else {
        return;
    };
    assert!(
        std::env::var_os("PADDOCK_MOE_DISK").is_none(),
        "the reference must run on the pinned backing"
    );
    let map = MappedGguf::open(&model).expect("open gguf");
    // Written before the pinned load, so the write never shares the host
    // with the pinned experts.
    if artifact.exists() {
        eprintln!(
            "artifact {} exists - kept, the writer did not run",
            artifact.display()
        );
    } else {
        let t = Instant::now();
        GpuQwen35::write_expert_artifact(&exec, &map, &artifact).expect("write artifact");
        let len = std::fs::metadata(&artifact).expect("artifact").len();
        eprintln!(
            "artifact written: {:.2} GiB in {:.1}s, peak RSS (VmHWM): {:?} KiB",
            len as f64 / (1u64 << 30) as f64,
            t.elapsed().as_secs_f64(),
            proc_field("/proc/self/status", "VmHWM:")
        );
    }
    drop_page_cache(&artifact);
    // SAFETY: as in setup.
    unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    let mut m = GpuQwen35::load(exec, &map, MAX_CTX).expect("load pinned");
    let slots = m.enable_moe_cache(u64::MAX).expect("seat expert cache");
    eprintln!("pinned: {slots} slots/layer");
    let (logits, tokens) = run(&mut m, &prompt(PARITY_PROMPT), PARITY_TOKENS);
    eprintln!("pinned tokens {tokens:?}");
    write_ref(&ref_path(&artifact), &logits, &tokens);
}

#[test]
fn moe_disk_parity_and_measure() {
    let Some((model, artifact, exec)) = setup() else {
        return;
    };
    assert!(
        std::env::var_os("PADDOCK_MOE_HOST").is_none(),
        "run with the disk backing only"
    );
    // SAFETY: as in setup.
    unsafe { std::env::set_var("PADDOCK_MOE_DISK", &artifact) };
    let (ref_logits, ref_tokens) = read_ref(&ref_path(&artifact));
    drop_page_cache(&artifact);
    let map = MappedGguf::open(&model).expect("open gguf");
    let t = Instant::now();
    let mut m = GpuQwen35::load(exec, &map, MAX_CTX).expect("load disk-backed");
    let slots = m.enable_moe_cache(u64::MAX).expect("seat expert cache");
    eprintln!(
        "disk: load {:.1}s, {slots} slots/layer",
        t.elapsed().as_secs_f64()
    );

    let (logits, tokens) = run(&mut m, &prompt(PARITY_PROMPT), PARITY_TOKENS);
    eprintln!("disk tokens   {tokens:?}");
    let mismatched = logits
        .iter()
        .zip(&ref_logits)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    eprintln!(
        "parity: {mismatched} of {} prefill logits differ bitwise, tokens equal={}",
        logits.len(),
        tokens == ref_tokens
    );

    assert_eq!(tokens, ref_tokens, "disk greedy tokens must equal pinned");
    assert_eq!(
        mismatched, 0,
        "disk prefill logits must equal pinned bit for bit"
    );
    measure(&mut m, Some(&artifact));
    let refused = m.enable_batch(1).err().map(|e| e.to_string());
    eprintln!("enable_batch on disk experts: {refused:?}");
    assert!(
        refused.is_some_and(|e| e.contains("batched serving")),
        "batched serving must refuse disk-backed experts"
    );
}

#[test]
fn moe_pinned_measure() {
    let Some((model, _, exec)) = setup() else {
        return;
    };
    // SAFETY: as in setup.
    unsafe { std::env::set_var("PADDOCK_MOE_HOST", "1") };
    let map = MappedGguf::open(&model).expect("open gguf");
    let mut m = GpuQwen35::load(exec, &map, MAX_CTX).expect("load pinned");
    let slots = m.enable_moe_cache(u64::MAX).expect("seat expert cache");
    eprintln!("pinned: {slots} slots/layer");
    measure(&mut m, None);
    if let Some((rows, misses)) = m.moe_cache_stats().expect("cache stats") {
        eprintln!(
            "pinned expert cache: {misses} misses of {rows} rows (hit {:.1}%)",
            100.0 * (1.0 - misses as f64 / rows.max(1) as f64)
        );
    }
}

#[test]
fn moe_disk_refuses_a_foreign_artifact() {
    let Some((model, _, exec)) = setup() else {
        return;
    };
    let bad = std::env::temp_dir().join(format!("pdk-foreign-{}.pdxa", std::process::id()));
    std::fs::write(&bad, vec![0x5au8; 1 << 16]).expect("write foreign file");
    // SAFETY: as in setup.
    unsafe { std::env::set_var("PADDOCK_MOE_DISK", &bad) };
    let map = MappedGguf::open(&model).expect("open gguf");
    let err = GpuQwen35::load(exec, &map, MAX_CTX)
        .err()
        .map(|e| e.to_string());
    eprintln!("foreign artifact: {err:?}");
    let _ = std::fs::remove_file(&bad);
    assert!(
        err.is_some_and(|e| e.contains("expert artifact") && e.contains("pdk-foreign")),
        "a foreign artifact must fail the load naming the file"
    );
}
