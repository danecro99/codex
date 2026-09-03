use std::path::PathBuf;
use std::sync::Arc;

use super::*;
use crate::context::world_state::WorldStateSnapshot;
use crate::context_manager::is_user_turn_boundary;
use codex_history::GuardianHistoryCheckpoint;
use codex_history::MATERIALIZED_RESUME_STATE_VERSION;
use codex_history::MaterializedResumeState;
use codex_history::ResponseItemEnvelope;
use codex_protocol::mcp::McpResourceOriginCheckpoint;
use codex_protocol::protocol::AgentStatus;
use codex_protocol::protocol::SessionContextWindow;
use codex_protocol::protocol::TokenUsageInfo;
use codex_protocol::protocol::TokenUsageRecord;
use codex_protocol::protocol::TruncationPolicy;
use uuid::Uuid;

const NEEDS_COMPACTION: &str = "codex_resume_state_needs_compaction";

/// Reconstructed model state and resume metadata produced by the canonical replay reducer.
#[derive(Debug, PartialEq)]
pub(super) struct RolloutReconstruction {
    pub(super) history: Arc<Vec<ResponseItemEnvelope>>,
    pub(super) guardian_history: Option<GuardianHistoryCheckpoint>,
    pub(super) previous_turn_settings: Option<PreviousTurnSettings>,
    pub(super) reference_context_item: Option<TurnContextItem>,
    pub(super) world_state_baseline: Option<WorldStateSnapshot>,
    pub(super) window_number: u64,
    pub(super) first_window_id: Option<Uuid>,
    pub(super) previous_window_id: Option<Uuid>,
    pub(super) window_id: Option<Uuid>,
    pub(super) token_info: Option<TokenUsageInfo>,
    pub(super) latest_token_usage_record: Option<TokenUsageRecord>,
    pub(super) last_agent_status: Option<AgentStatus>,
    pub(super) mcp_resource_origins: Option<McpResourceOriginCheckpoint>,
    pub(super) owned_startup_cwd: Option<PathBuf>,
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
    /// Full world-state snapshots are persisted after installing initial context. They still
    /// establish a baseline when a child fork removes the parent turn's agent message, so a
    /// segment that carries one after its latest compaction counts as a context baseline.
    has_full_world_state_since_compaction: bool,
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: TurnReferenceContextItem,
}

impl ActiveReplaySegment {
    fn accepts(&self, item_turn_id: Option<&str>) -> bool {
        self.turn_id
            .as_deref()
            .is_none_or(|turn_id| item_turn_id.is_none_or(|item_turn_id| item_turn_id == turn_id))
    }

    fn has_context_baseline(&self) -> bool {
        self.counts_as_user_turn || self.has_full_world_state_since_compaction
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
    latest_token_usage_record: Option<TokenUsageRecord>,
    last_agent_status: Option<AgentStatus>,
    mcp_resource_origins: Option<McpResourceOriginCheckpoint>,
    owned_startup_cwd: Option<PathBuf>,
    thread_id: ThreadId,
    truncation_policy: TruncationPolicy,
    checkpoint_suffix: bool,
    legacy_compaction_count: u64,
    saw_legacy_window: bool,
    saw_legacy_compaction_without_replacement_history: bool,
    auto_compact_window_prefill_input_tokens: Option<i64>,
    has_prior_user_turns: bool,
}

struct SeededResumeState {
    previous_turn_settings: Option<PreviousTurnSettings>,
    reference_context_item: Option<TurnContextItem>,
    world_state_baseline: Option<WorldStateSnapshot>,
    window: Option<ReconstructedWindow>,
    token_info: Option<TokenUsageInfo>,
    latest_token_usage_record: Option<TokenUsageRecord>,
    last_agent_status: Option<AgentStatus>,
    mcp_resource_origins: Option<McpResourceOriginCheckpoint>,
    owned_startup_cwd: Option<PathBuf>,
    auto_compact_window_prefill_input_tokens: Option<i64>,
    has_prior_user_turns: bool,
}

impl SeededResumeState {
    fn empty() -> Self {
        Self {
            previous_turn_settings: None,
            reference_context_item: None,
            world_state_baseline: None,
            window: None,
            token_info: None,
            latest_token_usage_record: None,
            last_agent_status: None,
            mcp_resource_origins: None,
            owned_startup_cwd: None,
            auto_compact_window_prefill_input_tokens: None,
            has_prior_user_turns: false,
        }
    }
}

impl ResumeReplayReducer {
    fn new(
        thread_id: ThreadId,
        truncation_policy: TruncationPolicy,
        materialized_state: Option<&MaterializedResumeState>,
    ) -> anyhow::Result<Self> {
        let mut history = ContextManager::new();
        let seeded = match materialized_state {
            Some(state) => Self::seed_from_materialized_state(&mut history, state)?,
            None => SeededResumeState::empty(),
        };
        Ok(Self {
            history,
            previous_turn_settings: seeded.previous_turn_settings,
            reference_context_item: seeded.reference_context_item,
            world_state_baseline: seeded.world_state_baseline,
            window: seeded.window,
            active_segment: None,
            token_info: seeded.token_info,
            latest_token_usage_record: seeded.latest_token_usage_record,
            last_agent_status: seeded.last_agent_status,
            mcp_resource_origins: seeded.mcp_resource_origins,
            owned_startup_cwd: seeded.owned_startup_cwd,
            thread_id,
            truncation_policy,
            checkpoint_suffix: materialized_state.is_some(),
            legacy_compaction_count: 0,
            saw_legacy_window: false,
            saw_legacy_compaction_without_replacement_history: false,
            auto_compact_window_prefill_input_tokens: seeded
                .auto_compact_window_prefill_input_tokens,
            has_prior_user_turns: seeded.has_prior_user_turns,
        })
    }

    fn seed_from_materialized_state(
        history: &mut ContextManager,
        state: &MaterializedResumeState,
    ) -> anyhow::Result<SeededResumeState> {
        if state.version != MATERIALIZED_RESUME_STATE_VERSION {
            anyhow::bail!(
                "{NEEDS_COMPACTION}: materialized state version {} is unsupported",
                state.version
            );
        }
        history.replace_annotated_arc(Arc::clone(&state.history));
        history.restore_guardian_history(state.guardian_history.as_ref());
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
        Ok(SeededResumeState {
            previous_turn_settings: state.previous_turn_settings.as_ref().map(|settings| {
                PreviousTurnSettings {
                    model: settings.model.clone(),
                    comp_hash: settings.comp_hash.clone(),
                    realtime_active: settings.realtime_active,
                }
            }),
            reference_context_item: state.reference_context_item.clone(),
            world_state_baseline,
            window: Some(ReconstructedWindow {
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
            token_info: state.token_info.clone(),
            latest_token_usage_record: state.latest_token_usage_record.clone(),
            last_agent_status: state.last_agent_status.clone(),
            mcp_resource_origins: state.mcp_resource_origins.clone(),
            owned_startup_cwd: state.owned_startup_cwd.clone(),
            auto_compact_window_prefill_input_tokens: state
                .auto_compact_window_prefill_input_tokens,
            has_prior_user_turns: state.has_prior_user_turns,
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
            RolloutItem::TokenUsageRecord(record) => {
                self.latest_token_usage_record = Some(record.clone());
            }
            RolloutItem::Compacted(compacted) => {
                if let Some(mcp_resource_origins) = &compacted.mcp_resource_origins {
                    self.mcp_resource_origins = Some(mcp_resource_origins.clone());
                }
                // A compaction snapshots the newest reachable usage record, so replay must not
                // keep an older one that the compaction already superseded.
                self.latest_token_usage_record = compacted.latest_token_usage_record.clone();
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
                    self.history
                        .restore_guardian_history(compacted.guardian_history.as_ref());
                    self.has_prior_user_turns = true;
                } else if self.checkpoint_suffix {
                    anyhow::bail!(
                        "{NEEDS_COMPACTION}: suffix contains a legacy compaction without replacement history"
                    );
                } else {
                    // Legacy rollouts without `replacement_history` should rebuild the historical
                    // TurnContext at the correct insertion point from persisted
                    // `TurnContextItem`s. These are rare enough that we currently just clear
                    // `reference_context_item`, reinject canonical context at the end of the
                    // resumed conversation, and accept the temporary out-of-distribution prompt
                    // shape.
                    self.saw_legacy_compaction_without_replacement_history = true;
                    let user_messages = crate::compact::collect_annotated_user_messages(
                        self.history.annotated_items(),
                    );
                    let rebuilt = crate::compact::build_compacted_history(
                        Vec::new(),
                        &user_messages,
                        &compacted.message,
                    );
                    self.history.replace_annotated(rebuilt);
                    self.has_prior_user_turns = true;
                }
                if let Some(active_segment) = self.active_segment.as_mut() {
                    active_segment.reference_context_item = TurnReferenceContextItem::Cleared;
                    active_segment.has_full_world_state_since_compaction = false;
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
                self.active_segment().has_full_world_state_since_compaction = true;
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
        let has_context_baseline = segment.has_context_baseline();
        if has_context_baseline && let Some(previous_turn_settings) = segment.previous_turn_settings
        {
            self.previous_turn_settings = Some(previous_turn_settings);
        }
        match segment.reference_context_item {
            TurnReferenceContextItem::NeverSet => {}
            TurnReferenceContextItem::Cleared => self.reference_context_item = None,
            TurnReferenceContextItem::Latest(item) if has_context_baseline => {
                self.reference_context_item = Some(*item);
            }
            TurnReferenceContextItem::Latest(_) => {}
        }
    }

    fn apply_event(&mut self, event: &EventMsg) {
        self.mcp_resource_origins =
            codex_mcp::reduce_resource_origin_checkpoint(self.mcp_resource_origins.as_ref(), event);
        // Copied or referenced history can carry another thread's settings, so only this thread's
        // own snapshots may claim its startup cwd.
        if let EventMsg::ThreadSettingsApplied(event) = event
            && event.thread_id == Some(self.thread_id)
        {
            self.owned_startup_cwd = Some(event.thread_settings.cwd.to_path_buf());
        }
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
        // A legacy compaction rebuilds history without the developer bundle that established the
        // stored baseline, so no later `TurnContextItem` can be trusted as a diff base.
        let reference_context_item = if self.saw_legacy_compaction_without_replacement_history {
            None
        } else {
            self.reference_context_item
        };
        RolloutReconstruction {
            guardian_history: self.history.guardian_history_checkpoint(),
            history: self.history.into_annotated_items_arc(),
            previous_turn_settings: self.previous_turn_settings,
            reference_context_item,
            world_state_baseline: self.world_state_baseline,
            window_number: window.number,
            first_window_id: window.first_id,
            previous_window_id: window.previous_id,
            window_id: window.id,
            token_info: self.token_info,
            latest_token_usage_record: self.latest_token_usage_record,
            last_agent_status: self.last_agent_status,
            mcp_resource_origins: self.mcp_resource_origins,
            owned_startup_cwd: self.owned_startup_cwd,
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
            | RolloutItem::TokenUsageRecord(_)
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
        if let Some(state) = materialized_state
            && state.truncation_policy != truncation_policy
        {
            anyhow::bail!(
                "{NEEDS_COMPACTION}: materialized state truncation policy does not match the active model"
            );
        }
        let replay_items = surviving_replay_items(rollout_items, materialized_state.is_some())?;
        let mut reducer =
            ResumeReplayReducer::new(self.thread_id(), truncation_policy, materialized_state)?;
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
