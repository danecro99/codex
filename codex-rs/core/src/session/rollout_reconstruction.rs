use std::sync::Arc;

use super::*;
use crate::context::world_state::WorldStateSnapshot;
use crate::context_manager::is_user_turn_boundary;
use codex_history::MATERIALIZED_RESUME_STATE_VERSION;
use codex_history::MaterializedResumeState;
use codex_history::ResponseItemEnvelope;
use codex_protocol::mcp::McpResourceOriginCheckpoint;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::SessionContextWindow;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::protocol::TruncationPolicy;
use uuid::Uuid;

const NEEDS_COMPACTION: &str = "codex_resume_state_needs_compaction";

/// Reconstructed model state and resume metadata produced by the canonical replay reducer.
#[derive(Debug, PartialEq)]
pub(super) struct RolloutReconstruction {
    pub(super) history: Arc<Vec<ResponseItemEnvelope>>,
    pub(super) previous_turn_settings: Option<PreviousTurnSettings>,
    pub(super) reference_context_item: Option<TurnContextItem>,
    pub(super) world_state_baseline: Option<WorldStateSnapshot>,
    pub(super) window_number: u64,
    pub(super) first_window_id: Option<Uuid>,
    pub(super) previous_window_id: Option<Uuid>,
    pub(super) window_id: Option<Uuid>,
    pub(super) token_info: Option<TokenUsageInfo>,
    pub(super) last_agent_status: Option<AgentStatus>,
    pub(super) mcp_resource_origins: Option<McpResourceOriginCheckpoint>,
    pub(super) auto_compact_window_prefill_input_tokens: Option<i64>,
    pub(super) has_prior_user_turns: bool,
}

#[derive(Debug, Clone, Copy)]
struct ReconstructedWindow {
    number: u64,
    first_id: Option<Uuid>,
    previous_id: Option<Uuid>,
    id: Option<Uuid>,
}

#[derive(Debug, Default)]
enum TurnReferenceContextItem {
    #[default]
    NeverSet,
    Cleared,
    Latest(Box<TurnContextItem>),
}

#[derive(Debug, Default)]
struct ActiveReplaySegment {
    turn_id: Option<String>,
    counts_as_user_turn: bool,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: TurnReferenceContextItem,
}

impl ActiveReplaySegment {
    fn accepts(&self, item_turn_id: Option<&str>) -> bool {
        self.turn_id
            .as_deref()
            .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
    }
}

struct ResumeReplayReducer {
    history: ContextManager,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: Option<TurnContextItem>,
    world_state_baseline: Option<WorldStateSnapshot>,
    window: Option<ReconstructedWindow>,
    active_segment: Option<ActiveReplaySegment>,
    token_info: Option<TokenUsageInfo>,
    last_agent_status: Option<AgentStatus>,
    mcp_resource_origins: Option<McpResourceOriginCheckpoint>,
    truncation_policy: TruncationPolicy,
    checkpoint_suffix: bool,
    legacy_compaction_count: u64,
    saw_legacy_window: bool,
    auto_compact_window_prefill_input_tokens: Option<i64>,
    has_prior_user_turns: bool,
}

impl ResumeReplayReducer {
    fn new(
        truncation_policy: TruncationPolicy,
        materialized_state: Option<&MaterializedResumeState>,
    ) -> anyhow::Result<Self> {
        let mut history = ContextManager::new();
        let (
            previous_turn_settings,
            reference_context_item,
            world_state_baseline,
            window,
            token_info,
            last_agent_status,
            mcp_resource_origins,
            auto_compact_window_prefill_input_tokens,
            has_prior_user_turns,
        ) = if let Some(state) = materialized_state {
            if state.version != MATERIALIZED_RESUME_STATE_VERSION {
                anyhow::bail!(
                    "{NEEDS_COMPACTION}: materialized state version {} is unsupported",
                    state.version
                );
            }
            if state.truncation_policy != truncation_policy {
                anyhow::bail!(
                    "{NEEDS_COMPACTION}: materialized state truncation policy does not match the active model"
                );
            }
            history.replace_annotated_arc(Arc::clone(&state.history));
            let world_state_baseline = state
                .world_state_baseline
                .as_ref()
                .map(|world_state| {
                    if !world_state.full {
                        anyhow::bail!(
                            "{NEEDS_COMPACTION}: materialized world-state baseline is not a full snapshot"
                        );
                    }
                    Ok(WorldStateSnapshot::from(&world_state.state))
                })
                .transpose()?;
            (
                state
                    .previous_turn_settings
                    .as_ref()
                    .map(|settings| PreviousTurnSettings {
                        model: settings.model.clone(),
                        comp_hash: settings.comp_hash.clone(),
                        realtime_active: settings.realtime_active,
                    }),
                state.reference_context_item.clone(),
                world_state_baseline,
                Some(ReconstructedWindow {
                    number: state.auto_compact_window.window_number,
                    first_id: Some(parse_required_uuid_v7(
                        &state.auto_compact_window.first_window_id,
                        "first window",
                    )?),
                    previous_id: state
                        .auto_compact_window
                        .previous_window_id
                        .as_deref()
                        .map(|value| parse_required_uuid_v7(value, "previous window"))
                        .transpose()?,
                    id: Some(parse_required_uuid_v7(
                        &state.auto_compact_window.window_id,
                        "current window",
                    )?),
                }),
                state.token_info.clone(),
                state.last_agent_status.clone(),
                state.mcp_resource_origins.clone(),
                state.auto_compact_window_prefill_input_tokens,
                state.has_prior_user_turns,
            )
        } else {
            (None, None, None, None, None, None, None, None, false)
        };
        Ok(Self {
            history,
            previous_turn_settings,
            reference_context_item,
            world_state_baseline,
            window,
            active_segment: None,
            token_info,
            last_agent_status,
            mcp_resource_origins,
            truncation_policy,
            checkpoint_suffix: materialized_state.is_some(),
            legacy_compaction_count: 0,
            saw_legacy_window: false,
            auto_compact_window_prefill_input_tokens,
            has_prior_user_turns,
        })
    }

    fn apply(&mut self, item: &RolloutItem) -> anyhow::Result<()> {
        match item {
            RolloutItem::SessionMeta(session_meta) => {
                if !self.checkpoint_suffix && self.window.is_none() {
                    self.window = session_meta
                        .meta
                        .context_window
                        .as_ref()
                        .and_then(reconstructed_window_from_session_context_window);
                }
            }
            RolloutItem::ResponseItem(response_item) => {
                let is_user_turn = is_user_turn_boundary(&response_item.item);
                self.active_segment().counts_as_user_turn |= is_user_turn;
                self.has_prior_user_turns |= is_user_turn;
                self.history.record_annotated_items(
                    std::slice::from_ref(response_item),
                    self.truncation_policy,
                );
            }
            RolloutItem::InterAgentCommunication(communication) => {
                self.active_segment().counts_as_user_turn = true;
                self.has_prior_user_turns = true;
                let response_item = communication.to_model_input_item();
                self.history
                    .record_items(std::iter::once(&response_item), self.truncation_policy);
            }
            RolloutItem::InterAgentCommunicationMetadata { .. } => {}
            RolloutItem::Compacted(compacted) => {
                if let Some(mcp_resource_origins) = &compacted.mcp_resource_origins {
                    self.mcp_resource_origins = Some(mcp_resource_origins.clone());
                }
                self.legacy_compaction_count = self.legacy_compaction_count.saturating_add(1);
                if compacted.window_number.is_none() {
                    self.saw_legacy_window = true;
                    if self.legacy_compaction_count == 1
                        && self.window.is_some_and(|window| window.number == 0)
                    {
                        self.window = None;
                    }
                    if self.checkpoint_suffix {
                        anyhow::bail!(
                            "{NEEDS_COMPACTION}: suffix contains a legacy compaction without a window number"
                        );
                    }
                }
                if let Some(replacement_history) = &compacted.replacement_history {
                    self.history.replace_annotated(replacement_history.clone());
                    self.has_prior_user_turns = true;
                } else {
                    anyhow::bail!(
                        "{NEEDS_COMPACTION}: rollout contains a legacy compaction without replacement history"
                    );
                }
                if let Some(active_segment) = self.active_segment.as_mut() {
                    active_segment.reference_context_item = TurnReferenceContextItem::Cleared;
                } else {
                    self.reference_context_item = None;
                }
                self.world_state_baseline = None;
                self.auto_compact_window_prefill_input_tokens = None;
                if let Some(window_number) = compacted.window_number {
                    self.window = Some(ReconstructedWindow {
                        number: window_number,
                        first_id: compacted.first_window_id.as_deref().and_then(parse_uuid_v7),
                        previous_id: compacted
                            .previous_window_id
                            .as_deref()
                            .and_then(parse_uuid_v7),
                        id: compacted.window_id.as_deref().and_then(parse_uuid_v7),
                    });
                }
            }
            RolloutItem::TurnContext(context) => {
                let active_segment = self.active_segment();
                if active_segment.turn_id.is_none() {
                    active_segment.turn_id = context.turn_id.clone();
                }
                if active_segment.accepts(context.turn_id.as_deref()) {
                    active_segment.previous_turn_settings = Some(PreviousTurnSettings {
                        model: context.model.clone(),
                        comp_hash: context.comp_hash.clone(),
                        realtime_active: context.realtime_active,
                    });
                    active_segment.reference_context_item =
                        TurnReferenceContextItem::Latest(Box::new(context.clone()));
                }
            }
            RolloutItem::WorldState(world_state) if world_state.full => {
                self.world_state_baseline = Some(WorldStateSnapshot::from(&world_state.state));
            }
            RolloutItem::WorldState(world_state) => {
                if let Some(baseline) = self.world_state_baseline.as_mut() {
                    baseline.apply_merge_patch(&world_state.state);
                } else if self.checkpoint_suffix {
                    anyhow::bail!(
                        "{NEEDS_COMPACTION}: suffix world-state patch has no full baseline"
                    );
                } else {
                    tracing::warn!("ignored world-state patch without a full snapshot");
                }
            }
            RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                self.finalize_active_segment();
                self.active_segment = Some(ActiveReplaySegment {
                    turn_id: Some(event.turn_id.clone()),
                    ..Default::default()
                });
                self.apply_event(&EventMsg::TurnStarted(event.clone()));
            }
            RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                let active_segment = self.active_segment();
                if active_segment.turn_id.is_none() {
                    active_segment.turn_id = Some(event.turn_id.clone());
                }
                if active_segment.accepts(Some(event.turn_id.as_str())) {
                    self.finalize_active_segment();
                }
                self.apply_event(&EventMsg::TurnComplete(event.clone()));
            }
            RolloutItem::EventMsg(EventMsg::TurnAborted(event)) => {
                let should_finalize = self.active_segment.as_ref().is_some_and(|segment| {
                    event
                        .turn_id
                        .as_deref()
                        .is_none_or(|turn_id| segment.accepts(Some(turn_id)))
                });
                if should_finalize {
                    self.finalize_active_segment();
                }
                self.apply_event(&EventMsg::TurnAborted(event.clone()));
            }
            RolloutItem::EventMsg(EventMsg::UserMessage(event)) => {
                self.active_segment().counts_as_user_turn = true;
                self.has_prior_user_turns = true;
                self.apply_event(&EventMsg::UserMessage(event.clone()));
            }
            RolloutItem::EventMsg(EventMsg::ThreadRolledBack(_)) => {
                unreachable!("rollback events are resolved before replay")
            }
            RolloutItem::EventMsg(event) => self.apply_event(event),
            RolloutItem::RealtimeItem(_) | RolloutItem::SecurityRiskScore(_) => {}
        }
        Ok(())
    }

    fn active_segment(&mut self) -> &mut ActiveReplaySegment {
        self.active_segment
            .get_or_insert_with(ActiveReplaySegment::default)
    }

    fn finalize_active_segment(&mut self) {
        let Some(segment) = self.active_segment.take() else {
            return;
        };
        if segment.counts_as_user_turn
            && let Some(previous_turn_settings) = segment.previous_turn_settings
        {
            self.previous_turn_settings = Some(previous_turn_settings);
        }
        match segment.reference_context_item {
            TurnReferenceContextItem::NeverSet => {}
            TurnReferenceContextItem::Cleared => self.reference_context_item = None,
            TurnReferenceContextItem::Latest(item) if segment.counts_as_user_turn => {
                self.reference_context_item = Some(*item);
            }
            TurnReferenceContextItem::Latest(_) => {}
        }
    }

    fn apply_event(&mut self, event: &EventMsg) {
        if let EventMsg::TokenCount(event) = event
            && let Some(info) = &event.info
        {
            self.token_info = Some(info.clone());
        }
        if let Some(status) = agent_status_from_event(event) {
            self.last_agent_status = Some(status);
        }
    }

    fn finish(mut self) -> RolloutReconstruction {
        self.finalize_active_segment();
        let window = self.window.unwrap_or(ReconstructedWindow {
            number: self.legacy_compaction_count,
            first_id: None,
            previous_id: None,
            id: None,
        });
        let window = if self.saw_legacy_window && window.number == 0 {
            ReconstructedWindow {
                number: self.legacy_compaction_count,
                first_id: window.first_id,
                previous_id: window.previous_id,
                id: window.id,
            }
        } else {
            window
        };
        RolloutReconstruction {
            history: self.history.into_annotated_items_arc(),
            previous_turn_settings: self.previous_turn_settings,
            reference_context_item: self.reference_context_item,
            world_state_baseline: self.world_state_baseline,
            window_number: window.number,
            first_window_id: window.first_id,
            previous_window_id: window.previous_id,
            window_id: window.id,
            token_info: self.token_info,
            last_agent_status: self.last_agent_status,
            mcp_resource_origins: self.mcp_resource_origins,
            auto_compact_window_prefill_input_tokens: self.auto_compact_window_prefill_input_tokens,
            has_prior_user_turns: self.has_prior_user_turns,
        }
    }
}

#[derive(Default)]
struct ReverseReplaySegment {
    indices: Vec<usize>,
    turn_id: Option<String>,
    counts_as_user_turn: bool,
}

impl ReverseReplaySegment {
    fn accepts(&self, item_turn_id: Option<&str>) -> bool {
        self.turn_id
            .as_deref()
            .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
    }
}

fn surviving_replay_items(
    rollout_items: &[RolloutItem],
    checkpoint_suffix: bool,
) -> anyhow::Result<Vec<&RolloutItem>> {
    let mut survives = vec![true; rollout_items.len()];
    let mut pending_rollback_turns = 0_usize;
    let mut active_segment: Option<ReverseReplaySegment> = None;

    for (index, item) in rollout_items.iter().enumerate().rev() {
        if let RolloutItem::EventMsg(EventMsg::ThreadRolledBack(rollback)) = item {
            survives[index] = false;
            if let Some(segment) = active_segment.take() {
                finalize_reverse_segment(
                    segment,
                    &mut pending_rollback_turns,
                    &mut survives,
                    /*drop_non_user_when_pending*/ true,
                );
            }
            pending_rollback_turns = pending_rollback_turns
                .saturating_add(usize::try_from(rollback.num_turns).unwrap_or(usize::MAX));
            continue;
        }
        if matches!(item, RolloutItem::SessionMeta(_)) {
            continue;
        }

        let segment = active_segment.get_or_insert_with(ReverseReplaySegment::default);
        segment.indices.push(index);
        match item {
            RolloutItem::EventMsg(EventMsg::TurnComplete(event)) => {
                if segment.turn_id.is_none() {
                    segment.turn_id = Some(event.turn_id.clone());
                }
            }
            RolloutItem::EventMsg(EventMsg::TurnAborted(event)) => {
                if segment.turn_id.is_none() {
                    segment.turn_id = event.turn_id.clone();
                }
            }
            RolloutItem::EventMsg(EventMsg::UserMessage(_)) => {
                segment.counts_as_user_turn = true;
            }
            RolloutItem::TurnContext(context) => {
                if segment.turn_id.is_none() {
                    segment.turn_id = context.turn_id.clone();
                }
            }
            RolloutItem::ResponseItem(response_item) => {
                if segment.turn_id.is_none()
                    && let Some(turn_id) = response_item.turn_id()
                {
                    segment.turn_id = Some(turn_id.to_string());
                }
                segment.counts_as_user_turn |= is_user_turn_boundary(&response_item.item);
                if is_user_turn_boundary(&response_item.item)
                    && segment.turn_id.is_none()
                    && let Some(segment) = active_segment.take()
                {
                    finalize_reverse_segment(
                        segment,
                        &mut pending_rollback_turns,
                        &mut survives,
                        /*drop_non_user_when_pending*/ true,
                    );
                }
            }
            RolloutItem::InterAgentCommunication(_) => segment.counts_as_user_turn = true,
            RolloutItem::EventMsg(EventMsg::TurnStarted(event)) => {
                if segment.accepts(Some(event.turn_id.as_str()))
                    && let Some(segment) = active_segment.take()
                {
                    finalize_reverse_segment(
                        segment,
                        &mut pending_rollback_turns,
                        &mut survives,
                        /*drop_non_user_when_pending*/ true,
                    );
                }
            }
            RolloutItem::SessionMeta(_)
            | RolloutItem::Compacted(_)
            | RolloutItem::WorldState(_)
            | RolloutItem::RealtimeItem(_)
            | RolloutItem::SecurityRiskScore(_)
            | RolloutItem::InterAgentCommunicationMetadata { .. }
            | RolloutItem::EventMsg(_) => {}
        }
    }
    if let Some(segment) = active_segment {
        finalize_reverse_segment(
            segment,
            &mut pending_rollback_turns,
            &mut survives,
            /*drop_non_user_when_pending*/ false,
        );
    }
    if checkpoint_suffix && pending_rollback_turns > 0 {
        anyhow::bail!(
            "{NEEDS_COMPACTION}: rollback crosses the materialized source fence by {pending_rollback_turns} user turns"
        );
    }
    Ok(rollout_items
        .iter()
        .zip(survives)
        .filter_map(|(item, survives)| survives.then_some(item))
        .collect())
}

fn finalize_reverse_segment(
    segment: ReverseReplaySegment,
    pending_rollback_turns: &mut usize,
    survives: &mut [bool],
    drop_non_user_when_pending: bool,
) {
    if *pending_rollback_turns == 0 || !segment.counts_as_user_turn && !drop_non_user_when_pending {
        return;
    }
    for index in segment.indices {
        survives[index] = false;
    }
    if segment.counts_as_user_turn {
        *pending_rollback_turns = pending_rollback_turns.saturating_sub(1);
    }
}

impl Session {
    pub(super) async fn reconstruct_resume_state(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
        materialized_state: Option<&MaterializedResumeState>,
    ) -> anyhow::Result<RolloutReconstruction> {
        let truncation_policy: TruncationPolicy =
            turn_context.model_info().truncation_policy.into();
        let replay_items = surviving_replay_items(rollout_items, materialized_state.is_some())?;
        let mut reducer = ResumeReplayReducer::new(truncation_policy, materialized_state)?;
        for item in replay_items {
            reducer.apply(item)?;
        }
        Ok(reducer.finish())
    }

    #[cfg(test)]
    pub(super) async fn reconstruct_history_from_rollout(
        &self,
        turn_context: &TurnContext,
        rollout_items: &[RolloutItem],
    ) -> RolloutReconstruction {
        self.reconstruct_resume_state(turn_context, rollout_items, None)
            .await
            .expect("full rollout reconstruction")
    }
}

fn parse_uuid_v7(value: &str) -> Option<Uuid> {
    Uuid::parse_str(value)
        .ok()
        .filter(|uuid| uuid.get_version_num() == 7)
}

fn parse_required_uuid_v7(value: &str, field: &str) -> anyhow::Result<Uuid> {
    parse_uuid_v7(value)
        .ok_or_else(|| anyhow::anyhow!("{NEEDS_COMPACTION}: materialized {field} ID is not UUIDv7"))
}

fn reconstructed_window_from_session_context_window(
    context_window: &SessionContextWindow,
) -> Option<ReconstructedWindow> {
    let id = parse_uuid_v7(&context_window.window_id)?;
    Some(ReconstructedWindow {
        number: 0,
        first_id: Some(id),
        previous_id: None,
        id: Some(id),
    })
}
