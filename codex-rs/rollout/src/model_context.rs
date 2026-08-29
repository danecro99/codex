use crate::ResponseItemEnvelope;
use crate::RolloutItem;
use codex_protocol::items::TurnItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::TruncationPolicy;
use std::error::Error;
use std::fmt;

use crate::reverse_jsonl_scanner::MAX_ROLLOUT_LINE_BYTES;

/// Maximum number of durable rollout items admitted to a resumed model context.
pub const MODEL_CONTEXT_MAX_ITEMS: usize = 16 * 1024;
/// Maximum estimated model tokens admitted from one durable rollout item.
pub const MODEL_CONTEXT_MAX_ITEM_TOKENS: usize = 10_000;
/// Maximum serialized bytes admitted to a resumed model context.
pub const MODEL_CONTEXT_MAX_BYTES: usize = MAX_ROLLOUT_LINE_BYTES;
/// Maximum estimated model tokens admitted to a resumed model context.
pub const MODEL_CONTEXT_MAX_TOKENS: usize = 1_000_000;

/// Whether a reverse model-context scan needs more rollout items.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelContextScanProgress {
    /// The reader should provide the next older rollout item.
    Continue,
    /// The scan has collected a safe bounded suffix.
    Complete,
}

/// Failure to prove or assemble a bounded model-context replay.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModelContextScanError {
    /// The scanned history did not contain a checkpoint that safely replaces its older prefix.
    MissingSafeCutoff,
    /// The candidate suffix exceeded one of the canonical resume limits.
    LimitExceeded {
        dimension: &'static str,
        actual: usize,
        maximum: usize,
    },
    /// A rollout item could not be measured before admission.
    Serialization(String),
}

impl fmt::Display for ModelContextScanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSafeCutoff => formatter
                .write_str("rollout does not contain a safe bounded model-context checkpoint"),
            Self::LimitExceeded {
                dimension,
                actual,
                maximum,
            } => write!(
                formatter,
                "bounded model context exceeds the {dimension} limit: {actual} > {maximum}"
            ),
            Self::Serialization(message) => {
                write!(
                    formatter,
                    "failed to measure bounded model context: {message}"
                )
            }
        }
    }
}

impl Error for ModelContextScanError {}

/// Accumulates newest-to-oldest rollout items until they are sufficient to reconstruct the latest
/// model context.
///
/// Storage implementations own how they fetch older items. Local JSONL readers and future
/// reverse-paged cloud readers can both feed their items through this scan to share the cutoff
/// rules and chronological replay assembly.
///
/// The scan stops once it has both:
///
/// - `saw_compaction`: a `CompactedItem` with `replacement_history` and `window_number`;
/// - `saw_completed_turn_context`: a completed user turn with a compatible `TurnContextItem`.
///
/// Reaching the beginning without this cutoff is an error for legacy resume. Callers that have
/// independently proved a canonical paginated lineage origin can finish through
/// [`Self::finish_paginated_origin`]. Callers with a separate, explicit frozen prefix boundary can
/// finish through [`Self::finish_frozen_prefix`].
///
/// `TurnContextItem` does not identify whether it came from a user turn, so one only counts after
/// the same turn also proves a user-turn boundary: a paginated
/// `ItemCompleted(UserMessage)` marker, agent message, or inter-agent message. Paginated writers
/// persist that marker for real user turns; older rollouts without it conservatively scan to the
/// beginning. A raw `role=user` response item is not sufficient because contextual user fragments
/// use that role but do not count as turn boundaries during reconstruction. The compaction restores
/// model-visible items; the turn context restores previous settings (`model`, `comp_hash`, and
/// `realtime_active`) and the reference baseline.
///
/// These paginated shapes disable the bounded cutoff:
///
/// - compaction without `replacement_history` or `window_number`;
/// - rollback markers;
///
/// When one appears, the scanner cannot produce a safe resume checkpoint.
#[derive(Debug, Default)]
pub struct ModelContextScan {
    items_newest_first: Vec<RolloutItem>,
    serialized_bytes: usize,
    estimated_tokens: usize,
    saw_compaction: bool,
    saw_completed_turn_context: bool,
    must_scan_to_start: bool,
    active_segment: ActiveTurnSegment,
}

impl ModelContextScan {
    /// Adds the next newest-to-oldest rollout item and reports whether the reader can stop.
    pub fn push(
        &mut self,
        item: RolloutItem,
    ) -> Result<ModelContextScanProgress, ModelContextScanError> {
        self.admit(&item)?;
        let progress = self.observe(&item);
        self.items_newest_first.push(item);
        Ok(progress)
    }

    /// Returns the collected items in chronological order with canonical head metadata.
    ///
    /// Call this after the reader reaches the beginning of its source or after [`Self::push`]
    /// returns [`ModelContextScanProgress::Complete`].
    pub fn finish(
        mut self,
        session_meta: SessionMetaLine,
    ) -> Result<Vec<RolloutItem>, ModelContextScanError> {
        if !self.has_bounded_cutoff() {
            return Err(ModelContextScanError::MissingSafeCutoff);
        }
        self.prepend_session_meta(session_meta)?;
        self.items_newest_first.reverse();
        Ok(self.items_newest_first)
    }

    /// Finishes a bounded scan that reached the canonical origin of a paginated lineage.
    ///
    /// The caller must prove the origin from the lineage rather than inferring it from an arbitrary
    /// `SessionMeta` item in replay history. Unsupported compaction and rollback shapes still fail
    /// because they cannot be reconstructed by this selector.
    pub fn finish_paginated_origin(
        mut self,
        session_meta: SessionMetaLine,
    ) -> Result<Vec<RolloutItem>, ModelContextScanError> {
        if self.must_scan_to_start {
            return Err(ModelContextScanError::MissingSafeCutoff);
        }
        self.prepend_session_meta(session_meta)?;
        self.items_newest_first.reverse();
        Ok(self.items_newest_first)
    }

    /// Finishes history whose older edge is already frozen by a durable `HistoryPosition`.
    pub fn finish_frozen_prefix(
        mut self,
        session_meta: SessionMetaLine,
    ) -> Result<Vec<RolloutItem>, ModelContextScanError> {
        self.items_newest_first.reverse();
        if !matches!(
            self.items_newest_first.first(),
            Some(RolloutItem::SessionMeta(_))
        ) {
            let item = RolloutItem::SessionMeta(session_meta);
            self.admit(&item)?;
            self.items_newest_first.insert(0, item);
        }
        Ok(self.items_newest_first)
    }

    fn prepend_session_meta(
        &mut self,
        session_meta: SessionMetaLine,
    ) -> Result<(), ModelContextScanError> {
        let item = RolloutItem::SessionMeta(session_meta);
        self.admit(&item)?;
        self.items_newest_first.push(item);
        Ok(())
    }

    fn admit(&mut self, item: &RolloutItem) -> Result<(), ModelContextScanError> {
        let item_bytes = serde_json::to_vec(item)
            .map_err(|err| ModelContextScanError::Serialization(err.to_string()))?
            .len();
        let item_tokens = TruncationPolicy::Bytes(item_bytes).token_budget();
        if item_tokens > MODEL_CONTEXT_MAX_ITEM_TOKENS {
            return Err(ModelContextScanError::LimitExceeded {
                dimension: "item token",
                actual: item_tokens,
                maximum: MODEL_CONTEXT_MAX_ITEM_TOKENS,
            });
        }
        let items = self.items_newest_first.len().saturating_add(1);
        let serialized_bytes = self.serialized_bytes.saturating_add(item_bytes);
        let estimated_tokens = self.estimated_tokens.saturating_add(item_tokens);
        for (dimension, actual, maximum) in [
            ("item", items, MODEL_CONTEXT_MAX_ITEMS),
            ("byte", serialized_bytes, MODEL_CONTEXT_MAX_BYTES),
            ("token", estimated_tokens, MODEL_CONTEXT_MAX_TOKENS),
        ] {
            if actual > maximum {
                return Err(ModelContextScanError::LimitExceeded {
                    dimension,
                    actual,
                    maximum,
                });
            }
        }
        self.serialized_bytes = serialized_bytes;
        self.estimated_tokens = estimated_tokens;
        Ok(())
    }

    fn observe(&mut self, item: &RolloutItem) -> ModelContextScanProgress {
        if self.must_scan_to_start {
            return ModelContextScanProgress::Continue;
        }

        match item {
            RolloutItem::Compacted(compacted)
                if compacted.replacement_history.is_none() || compacted.window_number.is_none() =>
            {
                self.must_scan_to_start = true;
            }
            RolloutItem::Compacted(_) => {
                self.saw_compaction = true;
            }
            RolloutItem::EventMsg(EventMsg::ThreadRolledBack(_)) => {
                // Paginated threads reject rollback. Keep old rollouts correct rather than
                // duplicating rollback survival semantics in this bounded selector.
                self.must_scan_to_start = true;
            }
            RolloutItem::EventMsg(EventMsg::ItemCompleted(event)) => {
                if self.active_segment.turn_id.is_none() {
                    self.active_segment.turn_id = Some(event.turn_id.clone());
                }
                if turn_ids_are_compatible(
                    self.active_segment.turn_id.as_deref(),
                    Some(event.turn_id.as_str()),
                ) {
                    self.active_segment.has_user_turn |=
                        matches!(&event.item, TurnItem::UserMessage(_));
                }
            }
            RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                self.active_segment
                    .turn_id
                    .get_or_insert_with(|| event.turn_id.clone());
            }
            RolloutItem::EventMsg(EventMsg::TurnAborted(event)) => {
                if let Some(turn_id) = &event.turn_id {
                    self.active_segment
                        .turn_id
                        .get_or_insert_with(|| turn_id.clone());
                }
            }
            RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                if turn_ids_are_compatible(
                    self.active_segment.turn_id.as_deref(),
                    Some(event.turn_id.as_str()),
                ) {
                    self.finalize_active_segment();
                }
            }
            RolloutItem::TurnContext(context) => {
                if self.active_segment.turn_id.is_none() {
                    self.active_segment.turn_id = context.turn_id.clone();
                }
                if turn_ids_are_compatible(
                    self.active_segment.turn_id.as_deref(),
                    context.turn_id.as_deref(),
                ) {
                    self.active_segment.has_turn_context = true;
                }
            }
            RolloutItem::ResponseItem(response_item) => {
                self.active_segment.has_user_turn |=
                    response_item_counts_as_user_turn(response_item);
            }
            RolloutItem::InterAgentCommunication(_) => {
                self.active_segment.has_user_turn = true;
            }
            RolloutItem::EventMsg(EventMsg::UserMessage(_)) => {
                self.active_segment.has_user_turn = true;
            }
            RolloutItem::EventMsg(_)
            | RolloutItem::SessionMeta(_)
            | RolloutItem::InterAgentCommunicationMetadata { .. }
            | RolloutItem::RealtimeItem(_)
            | RolloutItem::SecurityRiskScore(_)
            | RolloutItem::WorldState(_) => {}
        }

        if self.has_bounded_cutoff() {
            ModelContextScanProgress::Complete
        } else {
            ModelContextScanProgress::Continue
        }
    }

    fn finalize_active_segment(&mut self) {
        if self.active_segment.has_user_turn && self.active_segment.has_turn_context {
            self.saw_completed_turn_context = true;
        }
        self.active_segment = ActiveTurnSegment::default();
    }

    fn has_bounded_cutoff(&self) -> bool {
        !self.must_scan_to_start && self.saw_compaction && self.saw_completed_turn_context
    }
}

#[derive(Debug, Default)]
struct ActiveTurnSegment {
    turn_id: Option<String>,
    has_user_turn: bool,
    has_turn_context: bool,
}

fn turn_ids_are_compatible(active_turn_id: Option<&str>, item_turn_id: Option<&str>) -> bool {
    active_turn_id
        .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
}

fn response_item_counts_as_user_turn(response_item: &ResponseItemEnvelope) -> bool {
    match &response_item.item {
        ResponseItem::AgentMessage { .. } => true,
        ResponseItem::Message { role, content, .. } => {
            role == "assistant" && InterAgentCommunication::is_message_content(content)
        }
        _ => false,
    }
}
