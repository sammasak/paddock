//! Per-request event records + the bounded in-memory ring (doc §8.1).
//!
//! Every inference request produces exactly one record, built at the HTTP
//! edge (correlation ids, timing, status) and enriched by the handler (usage,
//! finish reasons, cached-prefix depth) through an [`EventScope`] riding the
//! request extensions. The record is pushed when the RESPONSE BODY completes -
//! a streaming completion's record carries its true duration and TTFT, and a
//! client disconnect still yields a record with whatever was known.
//!
//! Invariant (doc §8.7): observability never blocks serving. The hot-path cost
//! is one short mutex push into a fixed-size ring (~4K records ≈ 2 MB); there
//! are no per-token events and nothing ever touches disk. Subscribers read by
//! sequence number over the ADMIN surface; a reader that fell off the tail is
//! told "K events dropped" - never a silent gap.
//!
//! Field names follow the OTel GenAI semantic conventions where one exists
//! (`gen_ai.request.model`, `gen_ai.usage.*`, `gen_ai.response.finish_reasons`)
//! so the future OTLP export (§8.1) is a 1:1 mapping. Engine attributes with
//! no semconv equivalent are `paddock.*`: the per-request phase split
//! (tokenize at the edge; queue/prefill/decode measured inside the engine and
//! carried on `TokenEvent::Done` as `RunStats`), spec-decode drafted/accepted,
//! and the KV page footprint. `ttft_ms` runs from arrival at the edge to the
//! engine's first sampled token (tokenize + queue + prefill), not to the first
//! response byte: a stream's SSE preamble goes out before prefill finishes.
//! A chat, messages, completions or responses stream that ends before the
//! engine's stats arrive (a stop string, a disconnect, an error) ends it where
//! the handler received its first token; the responses agent loop does not
//! stamp one yet. Only a request without a token falls back to the
//! first byte, so every success still carries one TTFT sample.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

/// One request, one record (~0.5 KB serialized).
#[derive(Debug, Clone, Serialize)]
pub struct EventRecord {
    /// Ring sequence number - the subscriber's resume cursor.
    pub seq: u64,
    /// Unix millis at request arrival.
    pub ts_ms: u64,
    /// Request path (`/v1/chat/completions`, ...).
    pub endpoint: String,
    pub status: u16,
    /// Arrival -> response body complete (streaming included).
    pub duration_ms: u64,
    /// Whether the response went out as an event stream.
    pub stream: bool,
    /// Client `X-Request-ID`, honored and echoed; generated otherwise.
    pub request_id: String,
    /// Client `traceparent`, recorded verbatim so this request can join the
    /// caller's own distributed trace. Never propagated by us.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceparent: Option<String>,
    /// First matching session header (default `X-Session-ID`,
    /// `X-Litellm-Session-Id`) - explicit session identity for grouping.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// The OpenAI `user` / Anthropic `metadata.user_id` grouping key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    /// Served model id (the truthful one, not the request's alias).
    #[serde(
        rename = "gen_ai.request.model",
        skip_serializing_if = "Option::is_none"
    )]
    pub model: Option<String>,
    #[serde(
        rename = "gen_ai.usage.input_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub input_tokens: Option<u64>,
    #[serde(
        rename = "gen_ai.usage.output_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub output_tokens: Option<u64>,
    #[serde(
        rename = "gen_ai.response.finish_reasons",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub finish_reasons: Vec<String>,
    /// Prompt tokens served from the prefix cache = the engine's resume
    /// position for this request. The conversation-threading signal (§8.6).
    #[serde(
        rename = "paddock.prefix_resume_pos",
        skip_serializing_if = "Option::is_none"
    )]
    pub prefix_resume_pos: Option<u64>,
    /// Edge-measured prompt build: template render + tokenization. Multi-round
    /// requests (agent loops, n>1) sum their rounds - as do all phases below.
    #[serde(
        rename = "paddock.tokenize_ms",
        skip_serializing_if = "Option::is_none"
    )]
    pub tokenize_ms: Option<u64>,
    /// Engine-measured scheduler wait (submit -> slot admission).
    #[serde(rename = "paddock.queue_ms", skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    /// Engine-measured prefill wall clock (admission -> prompt prefilled).
    #[serde(rename = "paddock.prefill_ms", skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<u64>,
    /// Engine-measured decode wall clock (prefill done -> finish).
    #[serde(rename = "paddock.decode_ms", skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<u64>,
    /// Speculative tokens drafted / accepted across the request's rounds.
    /// Absent = the request never rode spec decode.
    #[serde(
        rename = "paddock.spec_drafted",
        skip_serializing_if = "Option::is_none"
    )]
    pub spec_drafted: Option<u64>,
    #[serde(
        rename = "paddock.spec_accepted",
        skip_serializing_if = "Option::is_none"
    )]
    pub spec_accepted: Option<u64>,
    /// Page-granular KV footprint at completion (max across rounds); absent
    /// on the serial (non-paged) engine.
    #[serde(rename = "paddock.kv_pages", skip_serializing_if = "Option::is_none")]
    pub kv_pages: Option<u64>,
    /// Arrival at the edge to the engine's first sampled token (tokenize +
    /// queue + prefill); a stream cut before the engine's stats arrived
    /// ends it at the handler's receipt of that token. A request without a
    /// token (embeddings, audio, an engine error or a disconnect before the
    /// first one) falls back to the first response-body byte. A completions
    /// stream that ends before sending any frame has no sample at all.
    #[serde(rename = "paddock.ttft_ms", skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    /// Output tokens over the post-first-token window - the edge view of decode
    /// rate. Streaming responses only (non-streamed compute is not separable
    /// at the edge).
    #[serde(
        rename = "paddock.decode_tok_s",
        skip_serializing_if = "Option::is_none"
    )]
    pub decode_tok_s: Option<f64>,
    /// FNV-1a of the presented bearer key - a stable grouping key, not a
    /// secret store (the key itself never lands in a record).
    #[serde(
        rename = "paddock.api_key_hash",
        skip_serializing_if = "Option::is_none"
    )]
    pub api_key_hash: Option<String>,
    /// Who sent it (`x-paddock-origin`: batch | studio); absent = live
    /// traffic. The forecaster's own-exhaust guard.
    #[serde(rename = "paddock.origin", skip_serializing_if = "Option::is_none")]
    pub origin: Option<&'static str>,
    /// The response body was dropped before it ran to end - the client
    /// vanished mid-stream. Without this bit an abandoned stream reads as a
    /// clean 200 with a short output count.
    #[serde(
        rename = "paddock.client_disconnected",
        skip_serializing_if = "is_false"
    )]
    pub client_disconnected: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// Handler-filled slots for the in-flight request. Cheap Mutex - a handful of
/// short writes per request, never on the token loop.
#[derive(Debug, Default)]
struct Cells {
    model: Option<String>,
    user: Option<String>,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    finish_reasons: Vec<String>,
    /// Accumulated engine RunStats (agent loops / n>1 sum their generations;
    /// kv_pages keeps the max - a footprint, not a flow).
    tokenize_ms: Option<u64>,
    queue_ms: Option<u64>,
    prefill_ms: Option<u64>,
    decode_ms: Option<u64>,
    spec_drafted: Option<u64>,
    spec_accepted: Option<u64>,
    kv_pages: Option<u64>,
    /// The first token: the earliest generation's `RunStats::prefilled_at`,
    /// or a stream handler's receipt of it when no Done arrived.
    first_token: Option<Instant>,
}

impl Cells {
    fn earliest_token(&mut self, t: Instant) {
        self.first_token = Some(self.first_token.map_or(t, |f| f.min(t)));
    }
}

/// Cloneable handle the middleware plants in the request extensions and the
/// handlers write through. Default (no middleware - unit tests, disabled
/// ring) is a no-op, so handlers call unconditionally.
#[derive(Debug, Clone, Default)]
pub struct EventScope(Option<Arc<Mutex<Cells>>>);

impl EventScope {
    fn live() -> Self {
        EventScope(Some(Arc::new(Mutex::new(Cells::default()))))
    }

    fn with(&self, f: impl FnOnce(&mut Cells)) {
        if let Some(c) = &self.0
            && let Ok(mut cells) = c.lock()
        {
            f(&mut cells);
        }
    }

    pub fn model(&self, id: &str) {
        self.with(|c| c.model = Some(id.to_owned()));
    }

    pub fn user(&self, u: Option<&str>) {
        if let Some(u) = u {
            self.with(|c| c.user = Some(u.to_owned()));
        }
    }

    /// Final token accounting (last write wins - agent loops report totals).
    pub fn usage(&self, input: usize, output: usize) {
        self.with(|c| {
            c.input_tokens = Some(input as u64);
            c.output_tokens = Some(output as u64);
        });
    }

    /// Prompt tokens served from the prefix cache (the resume position).
    pub fn cached(&self, n: usize) {
        self.with(|c| c.cached_tokens = Some(n as u64));
    }

    /// Edge prompt-build time: template render + tokenization. Agent loops
    /// re-prepare every round - rounds sum, like the engine phases.
    pub fn tokenized(&self, d: Duration) {
        self.with(|c| {
            c.tokenize_ms = Some(c.tokenize_ms.unwrap_or(0) + d.as_millis() as u64);
        });
    }

    /// Append one choice's finish reason.
    pub fn finish(&self, reason: &str) {
        self.with(|c| c.finish_reasons.push(reason.to_owned()));
    }

    /// Fold one generation's engine-measured RunStats into the record.
    /// Durations and spec counters SUM across a request's generations (agent
    /// rounds, n>1 choices) - total engine time attributable to the request;
    /// kv_pages keeps the MAX (a footprint, not a flow). Spec fields stay
    /// absent while drafted is 0, so "never rode spec" is distinguishable
    /// from "rode spec, nothing accepted".
    pub fn phases(&self, s: &paddock_engine::service::RunStats) {
        self.with(|c| {
            let add = |slot: &mut Option<u64>, v: u64| *slot = Some(slot.unwrap_or(0) + v);
            add(&mut c.queue_ms, s.queued_ms as u64);
            add(&mut c.prefill_ms, s.prefill_ms as u64);
            add(&mut c.decode_ms, s.decode_ms as u64);
            if s.spec_drafted > 0 {
                add(&mut c.spec_drafted, s.spec_drafted as u64);
                add(&mut c.spec_accepted, s.spec_accepted as u64);
            }
            if s.kv_pages > 0 {
                c.kv_pages = Some(c.kv_pages.unwrap_or(0).max(s.kv_pages as u64));
            }
            if let Some(t) = s.prefilled_at {
                c.earliest_token(t);
            }
        });
    }

    /// A streaming handler received a generation's first token. A stream
    /// that ends before the engine's Done (a stop string, a disconnect, an
    /// error) carries no `prefilled_at`, so this is its first token; when
    /// Done does arrive, the engine's earlier stamp wins.
    pub fn token_seen(&self) {
        let t = Instant::now();
        self.with(|c| c.earliest_token(t));
    }

    /// Public response timing, with no event metadata or credentials. Totals
    /// span all engine rounds, excluding tool execution and transport flushes.
    pub fn response_timing(&self) -> Option<serde_json::Value> {
        let c = self.0.as_ref()?.lock().ok()?;
        Some(serde_json::json!({
            "source": "engine", "version": 1,
            "queue_ms": c.queue_ms?, "prefill_ms": c.prefill_ms?,
            "decode_ms": c.decode_ms?,
        }))
    }

    fn take(&self) -> Cells {
        match &self.0 {
            Some(c) => c
                .lock()
                .map(|mut c| std::mem::take(&mut *c))
                .unwrap_or_default(),
            None => Cells::default(),
        }
    }
}

const RING_CAP: usize = 4096;

struct Inner {
    buf: VecDeque<Arc<EventRecord>>,
    /// Sequence the next record will get; oldest held = next_seq - buf.len().
    next_seq: u64,
}

/// The bounded in-memory ring. Fixed RAM, zero disk, overwrite-oldest.
pub struct EventRing {
    enabled: bool,
    inner: Mutex<Inner>,
    notify: tokio::sync::Notify,
    /// Cumulative records that fell off the tail (for the ops counter; a
    /// subscriber's per-read `dropped` is computed from sequences).
    overwritten: AtomicU64,
}

impl EventRing {
    pub fn new() -> Arc<Self> {
        Arc::new(EventRing {
            enabled: true,
            inner: Mutex::new(Inner {
                buf: VecDeque::with_capacity(RING_CAP),
                next_seq: 0,
            }),
            notify: tokio::sync::Notify::new(),
            overwritten: AtomicU64::new(0),
        })
    }

    /// `--no-events`: pushes are dropped, the admin endpoint reports disabled,
    /// identify omits the "events" capability.
    pub fn disabled() -> Arc<Self> {
        Arc::new(EventRing {
            enabled: false,
            inner: Mutex::new(Inner {
                buf: VecDeque::new(),
                next_seq: 0,
            }),
            notify: tokio::sync::Notify::new(),
            overwritten: AtomicU64::new(0),
        })
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    fn push(&self, mut record: EventRecord) {
        if !self.enabled {
            return;
        }
        {
            let Ok(mut inner) = self.inner.lock() else {
                return;
            };
            record.seq = inner.next_seq;
            inner.next_seq += 1;
            inner.buf.push_back(Arc::new(record));
            if inner.buf.len() > RING_CAP {
                inner.buf.pop_front();
                self.overwritten.fetch_add(1, Relaxed);
            }
        }
        self.notify.notify_waiters();
    }

    /// Records at sequence ≥ `from`, up to `max`. Returns
    /// (dropped-before-from, next-cursor, records) - `dropped > 0` means the
    /// reader fell off the ring's tail and lost exactly that many records.
    pub fn since(&self, from: u64, max: usize) -> (u64, u64, Vec<Arc<EventRecord>>) {
        let Ok(inner) = self.inner.lock() else {
            return (0, from, Vec::new());
        };
        let oldest = inner.next_seq - inner.buf.len() as u64;
        let dropped = oldest.saturating_sub(from);
        let start = from.max(oldest);
        let skip = (start - oldest) as usize;
        let out: Vec<Arc<EventRecord>> = inner.buf.iter().skip(skip).take(max).cloned().collect();
        let next = start + out.len() as u64;
        (dropped, next, out)
    }

    /// Long-poll form of [`since`]: waits up to `wait` for at least one record
    /// past `from` before returning (immediately when records already exist).
    pub async fn wait_since(
        &self,
        from: u64,
        max: usize,
        wait: Duration,
    ) -> (u64, u64, Vec<Arc<EventRecord>>) {
        let deadline = tokio::time::Instant::now() + wait;
        loop {
            // Arm the notification before checking, so a push between the check
            // and the await can't be missed.
            let notified = self.notify.notified();
            let (dropped, next, events) = self.since(from, max);
            if !events.is_empty() || dropped > 0 {
                return (dropped, next, events);
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return (dropped, next, events);
            }
            tokio::select! {
                _ = notified => {}
                _ = tokio::time::sleep_until(deadline) => {}
            }
        }
    }
}

/// FNV-1a 64 - stable across restarts (unlike SipHash's random keys), good
/// enough for a grouping key, deliberately not cryptographic.
fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Default session headers (llama-swap's default list - the LiteLLM/agent
/// ecosystem already sends these).
pub fn default_session_headers() -> Vec<String> {
    vec!["x-session-id".to_owned(), "x-litellm-session-id".to_owned()]
}

/// Everything the finalizer needs, carried by the response body.
struct Pending {
    ring: Arc<EventRing>,
    /// The second sink off this middleware. Independently gated:
    /// `--no-events` must not kill `/metrics`, nor `--no-metrics` the ring -
    /// both handles no-op internally when disabled.
    metrics: Arc<crate::metrics::Metrics>,
    scope: EventScope,
    endpoint: String,
    origin: crate::metrics::Origin,
    request_id: String,
    traceparent: Option<String>,
    session_id: Option<String>,
    api_key_hash: Option<String>,
    ts_ms: u64,
    t0: Instant,
    status: u16,
    stream: bool,
}

impl Pending {
    fn finalize(self, first_frame: Option<Duration>, completed: bool) {
        let duration = self.t0.elapsed();
        let cells = self.scope.take();
        let ttft = cells
            .first_token
            .map(|t| t.saturating_duration_since(self.t0))
            .or(first_frame);
        let ttft_ms = ttft.map(|d| d.as_millis() as u64);
        // Edge decode rate: tokens over the post-first-token window. Streaming
        // only - a non-streamed response holds every byte until the end.
        let decode_tok_s = match (self.stream, cells.output_tokens, ttft) {
            (true, Some(out), Some(t)) if out > 0 && duration > t => {
                Some(out as f64 / (duration - t).as_secs_f64())
            }
            _ => None,
        };
        self.metrics.observe(&crate::metrics::Observation {
            path: &self.endpoint,
            origin: self.origin,
            status: self.status,
            disconnected: !completed,
            model: cells.model.as_deref(),
            duration,
            ttft,
            decode_ms: cells.decode_ms,
            input_tokens: cells.input_tokens,
            output_tokens: cells.output_tokens,
            cached_tokens: cells.cached_tokens,
            spec_drafted: cells.spec_drafted,
            spec_accepted: cells.spec_accepted,
            trace: self
                .traceparent
                .as_deref()
                .and_then(crate::metrics::parse_traceparent),
        });
        self.ring.push(EventRecord {
            seq: 0, // assigned by push
            ts_ms: self.ts_ms,
            endpoint: self.endpoint,
            status: self.status,
            duration_ms: duration.as_millis() as u64,
            stream: self.stream,
            request_id: self.request_id,
            traceparent: self.traceparent,
            session_id: self.session_id,
            user: cells.user,
            model: cells.model,
            input_tokens: cells.input_tokens,
            output_tokens: cells.output_tokens,
            finish_reasons: cells.finish_reasons,
            prefix_resume_pos: cells.cached_tokens,
            tokenize_ms: cells.tokenize_ms,
            queue_ms: cells.queue_ms,
            prefill_ms: cells.prefill_ms,
            decode_ms: cells.decode_ms,
            spec_drafted: cells.spec_drafted,
            spec_accepted: cells.spec_accepted,
            kv_pages: cells.kv_pages,
            ttft_ms,
            decode_tok_s,
            api_key_hash: self.api_key_hash,
            origin: match self.origin {
                crate::metrics::Origin::Live => None,
                o => Some(match o {
                    crate::metrics::Origin::Batch => "batch",
                    _ => "studio",
                }),
            },
            client_disconnected: !completed,
        });
    }
}

/// Response body that timestamps its first data frame (the TTFT fallback for
/// a request without an engine token) and emits the event record when
/// dropped - i.e. when the body finished streaming, the client disconnected,
/// or the response was discarded.
struct RecordingBody {
    inner: axum::body::Body,
    pending: Option<Pending>,
    first_frame: Option<Duration>,
    /// The stream was polled to its natural end (`Ready(None)`). A body
    /// dropped without reaching it - and not already at end-of-stream - is a
    /// client that vanished mid-response: the `disconnect` accounting bit.
    completed: bool,
}

impl http_body::Body for RecordingBody {
    type Data = axum::body::Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Result<http_body::Frame<Self::Data>, Self::Error>>> {
        // axum's Body is Unpin (boxed inner stream) - safe projection.
        let this = self.get_mut();
        let poll = std::pin::Pin::new(&mut this.inner).poll_frame(cx);
        match &poll {
            std::task::Poll::Ready(None) => this.completed = true,
            std::task::Poll::Ready(Some(Ok(f))) => {
                if this.first_frame.is_none()
                    && f.is_data()
                    && let Some(p) = &this.pending
                {
                    this.first_frame = Some(p.t0.elapsed());
                }
            }
            _ => {}
        }
        poll
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

impl Drop for RecordingBody {
    fn drop(&mut self) {
        if let Some(p) = self.pending.take() {
            // A fixed-size body (plain JSON) may be finished off is_end_stream
            // without a final Ready(None) poll - that is a completion too;
            // only a drop with bytes still owed is a disconnect.
            let completed = self.completed || http_body::Body::is_end_stream(&self.inner);
            p.finalize(self.first_frame, completed);
        }
    }
}

/// A client-supplied request id we'll echo: visible ASCII, sane length.
fn valid_request_id(v: &axum::http::HeaderValue) -> Option<String> {
    let s = v.to_str().ok()?;
    (!s.is_empty() && s.len() <= 120 && s.bytes().all(|b| (0x21..=0x7e).contains(&b)))
        .then(|| s.to_owned())
}

/// Middleware: correlation + timing + the record push for inference paths.
/// Sits outside drain/auth so refused requests (503/401/429) get records too.
pub async fn events_mw(
    axum::extract::State(state): axum::extract::State<Arc<crate::routes::AppState>>,
    mut req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // Two sinks, two independent gates: `--no-events` must not silently kill
    // `/metrics`, nor `--no-metrics` the ring. Skip only when both are off.
    if (!state.events.enabled() && !state.metrics.enabled())
        || !crate::drain::is_inference_path(req.uri().path())
    {
        return next.run(req).await;
    }
    let t0 = Instant::now();
    let ts_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let headers = req.headers();
    let request_id = headers
        .get("x-request-id")
        .and_then(valid_request_id)
        .unwrap_or_else(|| format!("req_{}", uuid::Uuid::new_v4().simple()));
    let traceparent = headers
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let session_id = state.session_headers.iter().find_map(|h| {
        headers
            .get(h.as_str())
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned)
    });
    let api_key_hash = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|k| format!("{:016x}", fnv1a64(k)));
    let origin = crate::metrics::Origin::from_header(
        headers
            .get("x-paddock-origin")
            .and_then(|v| v.to_str().ok()),
    );
    let endpoint = req.uri().path().to_owned();

    let scope = EventScope::live();
    req.extensions_mut().insert(scope.clone());

    let mut res = next.run(req).await;

    if let Ok(v) = axum::http::HeaderValue::from_str(&request_id) {
        res.headers_mut().insert("x-request-id", v);
    }
    let stream = res
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|ct| ct.starts_with("text/event-stream"));
    let pending = Pending {
        ring: state.events.clone(),
        metrics: state.metrics.clone(),
        scope,
        endpoint,
        origin,
        request_id,
        traceparent,
        session_id,
        api_key_hash,
        ts_ms,
        t0,
        status: res.status().as_u16(),
        stream,
    };
    let (parts, body) = res.into_parts();
    axum::response::Response::from_parts(
        parts,
        axum::body::Body::new(RecordingBody {
            inner: body,
            pending: Some(pending),
            first_frame: None,
            completed: false,
        }),
    )
}

/// One request through the middleware's recording body, its response built
/// around the live scope, arriving at the edge at `t0` (None: once the
/// response is built, so setup time is not read as latency).
#[cfg(test)]
pub(crate) struct RecordingForTest {
    ring: Arc<EventRing>,
    body: RecordingBody,
}

#[cfg(test)]
impl RecordingForTest {
    pub(crate) fn new(
        metrics: Arc<crate::metrics::Metrics>,
        t0: Option<Instant>,
        body: impl FnOnce(EventScope) -> axum::body::Body,
    ) -> Self {
        let ring = EventRing::new();
        let scope = EventScope::live();
        scope.model("m1");
        let inner = body(scope.clone());
        let t0 = t0.unwrap_or_else(Instant::now);
        RecordingForTest {
            ring: ring.clone(),
            body: RecordingBody {
                inner,
                pending: Some(Pending {
                    ring,
                    metrics,
                    scope,
                    endpoint: "/v1/chat/completions".into(),
                    origin: Default::default(),
                    request_id: "req_t".into(),
                    traceparent: None,
                    session_id: None,
                    api_key_hash: None,
                    ts_ms: 0,
                    t0,
                    status: 200,
                    stream: true,
                }),
                first_frame: None,
                completed: false,
            },
        }
    }

    pub(crate) fn t0(&self) -> Instant {
        self.body.pending.as_ref().expect("not finished").t0
    }

    /// The next data frame as text, None at the end of the body.
    pub(crate) async fn frame(&mut self) -> Option<String> {
        use http_body_util::BodyExt as _;
        loop {
            let f = self.body.frame().await?.expect("infallible body");
            if let Ok(d) = f.into_data() {
                return Some(String::from_utf8_lossy(&d).into_owned());
            }
        }
    }

    /// Drop the body - at its end, or mid-stream as a client that left - and
    /// return the record it emitted.
    pub(crate) fn finish(self) -> Arc<EventRecord> {
        let RecordingForTest { ring, body } = self;
        drop(body);
        let (_, _, mut events) = ring.since(0, 1);
        events.pop().expect("one record")
    }
}

/// A stream whose SSE preamble is polled out first (header time), then each
/// generation's engine `stats` are folded in and the body ends. Returns the
/// record; the observation lands in `metrics`.
#[cfg(test)]
pub(crate) fn record_stream_for_test(
    metrics: Arc<crate::metrics::Metrics>,
    t0: Instant,
    cached: usize,
    stats: &[paddock_engine::service::RunStats],
) -> Arc<EventRecord> {
    let mut scope = EventScope::default();
    let mut r = RecordingForTest::new(metrics, Some(t0), |s| {
        s.usage(16, 4);
        s.cached(cached);
        scope = s;
        axum::body::Body::from("data: {\"role\":\"assistant\"}\n\n")
    });
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let first = http_body::Body::poll_frame(std::pin::Pin::new(&mut r.body), &mut cx);
    assert!(
        matches!(first, std::task::Poll::Ready(Some(Ok(_)))),
        "the preamble frame"
    );
    for s in stats {
        scope.phases(s);
    }
    while let std::task::Poll::Ready(Some(_)) =
        http_body::Body::poll_frame(std::pin::Pin::new(&mut r.body), &mut cx)
    {}
    r.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(endpoint: &str) -> EventRecord {
        EventRecord {
            seq: 0,
            ts_ms: 1,
            endpoint: endpoint.to_owned(),
            status: 200,
            duration_ms: 5,
            stream: false,
            request_id: "req_x".into(),
            traceparent: None,
            session_id: None,
            user: None,
            model: None,
            input_tokens: None,
            output_tokens: None,
            finish_reasons: Vec::new(),
            prefix_resume_pos: None,
            tokenize_ms: None,
            queue_ms: None,
            prefill_ms: None,
            decode_ms: None,
            spec_drafted: None,
            spec_accepted: None,
            kv_pages: None,
            ttft_ms: None,
            decode_tok_s: None,
            api_key_hash: None,
            origin: None,
            client_disconnected: false,
        }
    }

    #[test]
    fn ring_assigns_sequences_and_reports_drops() {
        let ring = EventRing::new();
        for i in 0..(RING_CAP + 10) {
            ring.push(record(&format!("/v1/x{i}")));
        }
        // Reader starting at 0 fell off the tail: exactly 10 dropped.
        let (dropped, next, events) = ring.since(0, 100);
        assert_eq!(dropped, 10);
        assert_eq!(events.first().unwrap().seq, 10);
        assert_eq!(next, 110);
        // Resume from the cursor: contiguous, no drops.
        let (dropped, next2, events) = ring.since(next, 100_000);
        assert_eq!(dropped, 0);
        assert_eq!(events.len(), RING_CAP - 100);
        assert_eq!(next2, (RING_CAP + 10) as u64);
        // At the head: nothing new.
        let (dropped, _, events) = ring.since(next2, 10);
        assert_eq!((dropped, events.len()), (0, 0));
    }

    #[test]
    fn disabled_ring_drops_everything() {
        let ring = EventRing::disabled();
        ring.push(record("/v1/chat/completions"));
        let (dropped, next, events) = ring.since(0, 10);
        assert_eq!((dropped, next, events.len()), (0, 0, 0));
    }

    #[tokio::test]
    async fn wait_since_returns_on_push() {
        let ring = EventRing::new();
        let r2 = ring.clone();
        let waiter =
            tokio::spawn(async move { r2.wait_since(0, 10, Duration::from_secs(5)).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        ring.push(record("/v1/messages"));
        let (dropped, next, events) = waiter.await.unwrap();
        assert_eq!((dropped, next, events.len()), (0, 1, 1));
        assert_eq!(events[0].endpoint, "/v1/messages");
    }

    #[test]
    fn semconv_names_on_the_wire() {
        let mut r = record("/v1/chat/completions");
        r.model = Some("qwen3.5-9b".into());
        r.input_tokens = Some(100);
        r.finish_reasons.push("stop".into());
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["gen_ai.request.model"], "qwen3.5-9b");
        assert_eq!(v["gen_ai.usage.input_tokens"], 100);
        assert_eq!(v["gen_ai.response.finish_reasons"][0], "stop");
        // Unset options stay off the wire (records stay ~0.5 KB).
        assert!(v.get("paddock.ttft_ms").is_none());
    }

    #[test]
    fn phases_sum_rounds_and_keep_kv_max() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        // two agent rounds: durations/spec sum, kv_pages keeps the max
        scope.phases(&paddock_engine::service::RunStats {
            sampled_stop: false,
            queued_ms: 5,
            prefill_ms: 100,
            decode_ms: 900,
            spec_drafted: 40,
            spec_accepted: 30,
            kv_pages: 12,
            prefilled_at: None,
        });
        scope.phases(&paddock_engine::service::RunStats {
            sampled_stop: false,
            queued_ms: 1,
            prefill_ms: 20,
            decode_ms: 400,
            spec_drafted: 10,
            spec_accepted: 8,
            kv_pages: 9,
            prefilled_at: None,
        });
        let timing = scope.response_timing().unwrap();
        assert_eq!(
            timing,
            serde_json::json!({
                "source":"engine", "version":1, "queue_ms":6, "prefill_ms":120, "decode_ms":1300
            })
        );
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/responses".into(),
            request_id: "req_p".into(),
            traceparent: None,
            session_id: None,
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let r = &events[0];
        assert_eq!(
            (r.queue_ms, r.prefill_ms, r.decode_ms),
            (Some(6), Some(120), Some(1300))
        );
        assert_eq!((r.spec_drafted, r.spec_accepted), (Some(50), Some(38)));
        assert_eq!(r.kv_pages, Some(12));
        let v = serde_json::to_value(r).unwrap();
        assert_eq!(v["paddock.prefill_ms"], 120);
        assert_eq!(v["paddock.spec_accepted"], 38);
        assert_eq!(v["paddock.kv_pages"], 12);
    }

    #[test]
    fn no_spec_stays_off_the_wire() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        // a request that never rode spec on the serial (non-paged) engine
        scope.phases(&paddock_engine::service::RunStats {
            sampled_stop: false,
            queued_ms: 0,
            prefill_ms: 10,
            decode_ms: 50,
            spec_drafted: 0,
            spec_accepted: 0,
            kv_pages: 0,
            prefilled_at: None,
        });
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/completions".into(),
            request_id: "req_q".into(),
            traceparent: None,
            session_id: None,
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let v = serde_json::to_value(&events[0]).unwrap();
        // "never rode spec" / "no paged pool" must be absent, not zero
        assert!(v.get("paddock.spec_drafted").is_none());
        assert!(v.get("paddock.kv_pages").is_none());
        assert_eq!(v["paddock.decode_ms"], 50);
    }

    #[test]
    fn scope_fills_flow_into_the_record() {
        let ring = EventRing::new();
        let scope = EventScope::live();
        scope.model("m");
        scope.usage(10, 5);
        scope.cached(8);
        scope.finish("stop");
        let p = Pending {
            ring: ring.clone(),
            metrics: crate::metrics::Metrics::disabled(),
            scope,
            origin: Default::default(),
            endpoint: "/v1/chat/completions".into(),
            request_id: "req_1".into(),
            traceparent: None,
            session_id: Some("s1".into()),
            api_key_hash: None,
            ts_ms: 0,
            t0: Instant::now(),
            status: 200,
            stream: false,
        };
        p.finalize(None, true);
        let (_, _, events) = ring.since(0, 10);
        let r = &events[0];
        assert_eq!(r.model.as_deref(), Some("m"));
        assert_eq!((r.input_tokens, r.output_tokens), (Some(10), Some(5)));
        assert_eq!(r.prefix_resume_pos, Some(8));
        assert_eq!(r.session_id.as_deref(), Some("s1"));
        assert_eq!(r.finish_reasons, ["stop"]);
    }

    /// The streaming endpoints that stamp their first token's receipt.
    #[derive(Clone, Copy, Debug)]
    enum Endpoint {
        Chat,
        Messages,
        Completions,
        Responses,
    }
    const ENDPOINTS: [Endpoint; 4] = [
        Endpoint::Chat,
        Endpoint::Messages,
        Endpoint::Completions,
        Endpoint::Responses,
    ];
    /// The endpoints with a client `stop` parameter (responses has none).
    const STOP_ENDPOINTS: [Endpoint; 3] =
        [Endpoint::Chat, Endpoint::Messages, Endpoint::Completions];

    /// What the stop-string exit records as its finish reason.
    fn stop_reason(e: Endpoint) -> &'static str {
        match e {
            Endpoint::Messages => "stop_sequence",
            Endpoint::Chat | Endpoint::Completions | Endpoint::Responses => "stop",
        }
    }

    type TokenTx = tokio::sync::mpsc::UnboundedSender<paddock_engine::service::TokenEvent>;

    /// The gap the tests open between instants they compare. Sleeping only
    /// bounds a gap from below, so every bound is a captured instant and a
    /// loaded host stretches the gaps without breaking an assertion.
    const DELAY: Duration = Duration::from_millis(20);

    /// One streamed answer through `e`'s real handler and the recording body.
    struct Stream {
        tx: TokenTx,
        r: RecordingForTest,
        marks: Marks,
    }

    /// The instants a stream's TTFT is checked against.
    #[derive(Clone, Copy)]
    struct Marks {
        t0: Instant,
        /// Taken once the preamble is out: a first-byte fallback is no later.
        header: Instant,
    }

    impl Stream {
        async fn new(e: Endpoint, stop: &[&str]) -> Self {
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
            let metrics = crate::metrics::Metrics::new("i".into(), Default::default(), None);
            let mut r = RecordingForTest::new(metrics, None, |scope| match e {
                Endpoint::Chat => crate::chat::stream_for_test(scope, stop, vec![rx]),
                Endpoint::Messages => crate::messages::stream_for_test(scope, stop, rx),
                Endpoint::Completions => crate::completions::stream_for_test(scope, stop, vec![rx]),
                Endpoint::Responses => crate::responses::stream_for_test(scope, rx),
            });
            let t0 = r.t0();
            // chat's role delta, message_start and ping, response.created and
            // in_progress; completions sends none
            idle(&mut r).await;
            let header = Instant::now();
            Stream {
                tx,
                r,
                marks: Marks { t0, header },
            }
        }

        /// After `DELAY`, a first token that decodes to no text yet (half of
        /// `é`), so no endpoint sends a frame for it. Returns the instants
        /// around its receipt: sent, and seen once the handler went idle.
        async fn first_token(&mut self) -> (Instant, Instant) {
            tokio::time::sleep(DELAY).await;
            let sent = Instant::now();
            self.tx.send(token(8)).unwrap();
            idle(&mut self.r).await;
            let seen = Instant::now();
            // the exit's frames come `DELAY` after the receipt, so a
            // first-frame fallback taken there cannot pass for it
            tokio::time::sleep(DELAY).await;
            (sent, seen)
        }

        /// Drop the body (a client that left, if it had not ended) and
        /// return its record.
        fn finish(self) -> (Arc<EventRecord>, Marks) {
            (self.r.finish(), self.marks)
        }
    }

    impl Marks {
        fn ms(self, t: Instant) -> u64 {
            t.duration_since(self.t0).as_millis() as u64
        }

        /// The TTFT lies within the first token's receipt, past the header.
        fn assert_first_token(
            self,
            e: Endpoint,
            rec: &EventRecord,
            (sent, seen): (Instant, Instant),
        ) {
            let ttft = rec
                .ttft_ms
                .unwrap_or_else(|| panic!("{e:?}: no TTFT sample"));
            assert!(
                self.ms(sent) > self.ms(self.header),
                "{e:?}: DELAY opens the gap"
            );
            assert!(
                (self.ms(sent)..=self.ms(seen)).contains(&ttft),
                "{e:?}: {ttft} ms, the token arrived in {}..={} ms",
                self.ms(sent),
                self.ms(seen)
            );
            assert!(ttft <= rec.duration_ms, "{e:?}");
        }

        /// The documented first-byte fallback: no later than the header for
        /// an endpoint with a preamble. Completions has none, so a stream
        /// without a frame has no sample.
        fn assert_first_byte(self, e: Endpoint, rec: &EventRecord, framed: bool) {
            match (e, framed) {
                (Endpoint::Completions, false) => assert_eq!(rec.ttft_ms, None, "{e:?}"),
                (Endpoint::Completions, true) => {
                    assert!(
                        rec.ttft_ms.is_some_and(|ms| ms > self.ms(self.header)),
                        "{e:?}"
                    )
                }
                _ => assert!(
                    rec.ttft_ms.is_some_and(|ms| ms <= self.ms(self.header)),
                    "{e:?}: {:?} ms, header at {} ms",
                    rec.ttft_ms,
                    self.ms(self.header)
                ),
            }
        }
    }

    /// Read frames until the handler has gone `DELAY` without one.
    async fn idle(r: &mut RecordingForTest) {
        while let Ok(Some(_)) = tokio::time::timeout(DELAY, r.frame()).await {}
    }

    fn token(id: u32) -> paddock_engine::service::TokenEvent {
        paddock_engine::service::TokenEvent::Token { id, logprobs: None }
    }

    fn engine_error() -> paddock_engine::service::TokenEvent {
        paddock_engine::service::TokenEvent::Error(paddock_engine::service::EngineError::internal(
            "boom",
        ))
    }

    async fn drain(r: &mut RecordingForTest) -> Vec<String> {
        let mut frames = Vec::new();
        while let Some(f) = r.frame().await {
            frames.push(f);
        }
        frames
    }

    #[tokio::test]
    async fn a_stop_string_exit_ends_ttft_at_the_first_token_not_the_header() {
        for e in STOP_ENDPOINTS {
            let mut s = Stream::new(e, &["ll"]).await;
            let at = s.first_token().await;
            // "é", "he" then "ll": the stop string hits and the handler leaves
            // without waiting for the engine's Done
            for id in [9, 4, 5] {
                s.tx.send(token(id)).unwrap();
            }
            let frames = drain(&mut s.r).await;
            assert!(frames.iter().any(|f| f.contains("é")), "{e:?}: {frames:?}");
            let (rec, marks) = s.finish();
            assert!(!rec.client_disconnected, "{e:?}");
            assert_eq!(rec.finish_reasons, [stop_reason(e)], "{e:?}");
            marks.assert_first_token(e, &rec, at);
        }
    }

    #[tokio::test]
    async fn a_disconnect_after_the_first_token_keeps_its_arrival() {
        for e in ENDPOINTS {
            let mut s = Stream::new(e, &[]).await;
            let at = s.first_token().await;
            let (rec, marks) = s.finish();
            assert!(rec.client_disconnected, "{e:?}");
            marks.assert_first_token(e, &rec, at);
        }
    }

    #[tokio::test]
    async fn an_engine_error_after_the_first_token_keeps_its_arrival() {
        for e in ENDPOINTS {
            let mut s = Stream::new(e, &[]).await;
            let at = s.first_token().await;
            s.tx.send(engine_error()).unwrap();
            drain(&mut s.r).await;
            let (rec, marks) = s.finish();
            marks.assert_first_token(e, &rec, at);
        }
    }

    /// No token at all: the documented first-byte fallback, at most one
    /// sample per request, no panic - on an engine error and on a client
    /// that left.
    #[tokio::test]
    async fn a_stream_without_a_token_falls_back_to_its_first_byte() {
        for e in ENDPOINTS {
            let mut s = Stream::new(e, &["ll"]).await;
            tokio::time::sleep(DELAY).await;
            s.tx.send(engine_error()).unwrap();
            assert!(!drain(&mut s.r).await.is_empty(), "{e:?}");
            let (rec, marks) = s.finish();
            marks.assert_first_byte(e, &rec, true);

            let (rec, marks) = Stream::new(e, &[]).await.finish();
            assert!(rec.client_disconnected, "{e:?}");
            marks.assert_first_byte(e, &rec, false);
        }
    }

    /// The endpoints with an agent loop, and the preamble frame each sends
    /// before its first round.
    const AGENT_LOOPS: [(Endpoint, &str); 2] = [
        (Endpoint::Messages, "event: message_start\n\n"),
        (Endpoint::Responses, "event: response.created\n\n"),
    ];

    /// An agent loop's recorded stream with its preamble out. The loops have
    /// no CPU-only stream (each round submits to the model's engine), so a
    /// test drives the receipts their Token arms make through
    /// `messages::agent_token`, and their rounds' Done through `phases`.
    struct AgentLoop {
        r: RecordingForTest,
        scope: EventScope,
        marks: Marks,
        metrics: Arc<crate::metrics::Metrics>,
    }

    impl AgentLoop {
        async fn new(preamble: &'static str) -> Self {
            let metrics = crate::metrics::Metrics::new("i".into(), Default::default(), None);
            let mut scope = EventScope::default();
            let mut r = RecordingForTest::new(metrics.clone(), None, |s| {
                scope = s;
                axum::body::Body::from(preamble)
            });
            let t0 = r.t0();
            r.frame().await.expect("the preamble");
            let marks = Marks {
                t0,
                header: Instant::now(),
            };
            AgentLoop {
                r,
                scope,
                marks,
                metrics,
            }
        }

        /// One round's token receipts into a fresh `ids`, as the round's
        /// Token arm makes them.
        fn round(&self, tokens: &[u32]) {
            let mut ids = Vec::new();
            for &t in tokens {
                crate::messages::agent_token(&self.scope, &mut ids, t);
            }
            assert_eq!(ids, tokens);
        }

        /// A round's Done, its engine first-token stamp `at`.
        fn done(&self, at: Instant) {
            self.scope.phases(&paddock_engine::service::RunStats {
                prefilled_at: Some(at),
                ..Default::default()
            });
        }

        /// Drop the body (a client that left) and return its record, after
        /// checking the request observed exactly one TTFT sample.
        fn finish(self, e: Endpoint) -> (Arc<EventRecord>, Marks) {
            let rec = self.r.finish();
            let s = self.metrics.render(crate::metrics::Format::Classic, 0);
            let samples: u64 = s
                .lines()
                .filter(|l| l.starts_with("gen_ai_server_time_to_first_token_seconds_count"))
                .filter_map(|l| l.rsplit(' ').next()?.parse::<u64>().ok())
                .sum();
            assert_eq!(samples, 1, "{e:?}: {s}");
            (rec, self.marks)
        }
    }

    /// Rounds cut before their Done: round 0's first token is the TTFT, past
    /// the header, and neither round 0's later tokens nor a later round's
    /// receipts or engine stamp (hand-set well after it) move it.
    #[tokio::test]
    async fn an_agent_loop_keeps_its_first_rounds_token() {
        for (e, preamble) in AGENT_LOOPS {
            let a = AgentLoop::new(preamble).await;
            tokio::time::sleep(DELAY).await;
            let sent = Instant::now();
            a.round(&[8]);
            let seen = Instant::now();
            tokio::time::sleep(DELAY).await;
            a.round(&[4, 5]);
            a.done(seen + Duration::from_secs(1));
            let (rec, marks) = a.finish(e);
            marks.assert_first_token(e, &rec, (sent, seen));
        }
    }

    /// Round 0 reached its Done: its engine stamp (hand-set) stands, and a
    /// later round cut before its Done does not displace it.
    #[tokio::test]
    async fn an_agent_loop_keeps_round_zeros_engine_stamp() {
        for (e, preamble) in AGENT_LOOPS {
            let a = AgentLoop::new(preamble).await;
            // the handler's receipt trails the engine's stamp
            tokio::time::sleep(DELAY).await;
            a.round(&[8, 9]);
            a.done(a.marks.t0 + Duration::from_millis(1));
            a.round(&[4, 5]);
            let (rec, _) = a.finish(e);
            assert_eq!(rec.ttft_ms, Some(1), "{e:?}");
        }
    }

    /// A loop that ends without a token (an engine error, a client that left
    /// during prefill): the documented first-byte fallback, no panic.
    #[tokio::test]
    async fn an_agent_loop_without_a_token_falls_back_to_its_first_byte() {
        for (e, preamble) in AGENT_LOOPS {
            let a = AgentLoop::new(preamble).await;
            a.scope.phases(&Default::default());
            let (rec, marks) = a.finish(e);
            marks.assert_first_byte(e, &rec, true);
        }
    }

    /// The handler's receipt never displaces the engine's earlier stamp,
    /// hand-set here so no clock reading decides the result.
    #[tokio::test]
    async fn the_done_path_keeps_the_engine_first_token() {
        for e in ENDPOINTS {
            let mut s = Stream::new(e, &[]).await;
            s.first_token().await;
            s.tx.send(paddock_engine::service::TokenEvent::Done(
                paddock_engine::service::FinishReason::Length,
                paddock_engine::service::RunStats {
                    prefilled_at: Some(s.marks.t0 + Duration::from_millis(1)),
                    ..Default::default()
                },
            ))
            .unwrap();
            drain(&mut s.r).await;
            let (rec, _) = s.finish();
            assert_eq!(rec.ttft_ms, Some(1), "{e:?}");
        }
    }
}
