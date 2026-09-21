# Thread Store

`codex-thread-store` is the storage boundary for Codex threads. It defines the
`ThreadStore` trait plus local and in-memory implementations. Other storage
implementations may live outside this repository.

## Responsibilities

- `ThreadStore::append_items` is the raw canonical history append API. It does
  not infer metadata from item contents.
- `ThreadStore::update_thread_metadata` is the only thread metadata write API.
  It accepts a single literal metadata patch shape, regardless of whether the
  caller is applying a user/API mutation or facts derived above the store from
  appended history.
- `LiveThread` is the preferred API for active session persistence. It owns a
  per-thread metadata sync helper, applies the rollout persistence policy,
  appends canonical history, and then sends metadata patches through
  `ThreadStore::update_thread_metadata`.
- `ThreadManager` routes metadata mutations for loaded and cold threads through
  one entrypoint. Loaded threads use their `LiveThread`; cold threads go
  directly to the store.
- `LocalThreadStore` persists history through `codex-rollout` JSONL files and
  persists queryable metadata through the SQLite state database when available.
  Local explicit metadata mutations also maintain JSONL/name-index compatibility
  so reading old or SQLite-less local storage keeps working.
- `RolloutRecorder` is the local JSONL writer. It writes already-canonical
  items for `ThreadStore::append_items`; it no longer decides metadata updates
  for live thread-store appends.
- `core/session` creates or resumes `LiveThread` handles and does not need to
  know whether persistence is backed by local files or another store.

## Direction

New metadata observation semantics should live above `ThreadStore`. Stores
persist explicit metadata fields, but raw history appends remain history-only.

## Private resume state in the 0.155.1 integration

`MaterializedResumeState` version 7 stores the upstream `RetainedContext` alongside
model history and the remaining exact resume state. The canonical source is still
the rollout, including retained-context events and compaction snapshots. Both full
reconstruction and checkpoint-plus-suffix reconstruction must produce the same
retained review evidence. Compaction persists this evidence with its checkpoint
before replacing live session history.

Reconstruction keeps upstream's two distinct boundaries: surviving provider-turn
segments determine resume metadata and the newest surviving compaction; model
history is folded from that checkpoint's original suffix through `ContextManager`,
including its instruction-level rollback operation. A later steer in the same
provider turn must not erase an earlier retained answer. Reverse-filtering those
source items before the history fold would lose that evidence. A materialized
checkpoint seeds the same fold, not a second transcript or an alternative reader.

The current derived-state namespace is `materialized_resume_state_v7`. No previous
namespace is read, rewritten, or treated as version 7. The existing first-use-miss
path reconstructs from the canonical transcript and publishes a newly fenced
version 7 checkpoint. Diagnostics report `Miss` for this first reconstruction;
subsequent matching resumes can report `Hit`. Invalid data in the current namespace
is an error, not permission to silently discard it and retry through another path.
Original transcripts and append-generation journals are not migrated or truncated.

The resume writer uses the upstream canonical reverse JSON decoder. For paginated
history it must read the actual terminal record, including floating-point token
counts, and assign the next ordinal. An invalid terminal record is refused before
even newline repair; it is never skipped in favor of an older valid ordinal.
Compaction cannot repair an already stopped writer. That runtime must only be
replaced after preserving its available evidence; unsaved in-memory conversation
content is not falsely promised as durable history.
