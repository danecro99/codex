use super::*;

use super::tests::build_world_state_from_turn_context;
use super::tests::make_session_and_context;
use super::tests::raw_history_items;
use crate::context::ContextualUserFragment;
use codex_history::CompactedItem;
use codex_history::InitialHistory;
use codex_history::MATERIALIZED_RESUME_STATE_VERSION;
use codex_history::MaterializedAutoCompactWindow;
use codex_history::MaterializedPreviousTurnSettings;
use codex_history::MaterializedResumeState;
use codex_history::ResponseItemEnvelope;
use codex_history::ResumedHistory;
use codex_protocol::AgentPath;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::InterAgentCommunication;
use codex_protocol::protocol::SessionContextWindow;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::TruncationPolicy;
use codex_protocol::protocol::WorldStateItem;
use codex_protocol::security_risk::SecurityRiskScore;
use core_test_support::responses::strip_metadata_from_items;
use pretty_assertions::assert_eq;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;
use uuid::Uuid;

macro_rules! object {
    ($value:tt) => {
        serde_json::from_value(json!($value)).unwrap()
    };
}

fn user_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "user".to_string(),
        content: vec![ContentItem::InputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn assistant_message(text: &str) -> ResponseItem {
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: text.to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn turn_started(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnStarted(
        codex_protocol::protocol::TurnStartedEvent {
            turn_id: turn_id.to_string(),
            trace_id: None,
            started_at: None,
            model_context_window: Some(128_000),
            collaboration_mode_kind: ModeKind::Default,
        },
    ))
}

fn turn_complete(turn_id: &str) -> RolloutItem {
    RolloutItem::EventMsg(EventMsg::TurnComplete(
        codex_protocol::protocol::TurnCompleteEvent {
            turn_id: turn_id.to_string(),
            last_agent_message: None,
            error: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            time_to_first_token_ms: None,
        },
    ))
}

fn annotated(items: Vec<ResponseItem>) -> Vec<ResponseItemEnvelope> {
    items.into_iter().map(ResponseItemEnvelope::new).collect()
}

fn inter_agent_assistant_message(text: &str) -> ResponseItem {
    let communication = InterAgentCommunication::new(
        AgentPath::root(),
        AgentPath::root().join("worker").unwrap(),
        Vec::new(),
        text.to_string(),
        /*trigger_turn*/ true,
    );
    ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: serde_json::to_string(&communication).unwrap(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    }
}

fn completed_user_turn_rollout(
    turn_context_item: TurnContextItem,
    items: Vec<RolloutItem>,
) -> Vec<RolloutItem> {
    let turn_id = turn_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let mut rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(turn_context_item),
    ];
    rollout_items.extend(items);
    rollout_items.push(RolloutItem::EventMsg(EventMsg::TurnComplete(
        codex_protocol::protocol::TurnCompleteEvent {
            turn_id,
            last_agent_message: None,
            error: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            time_to_first_token_ms: None,
        },
    )));
    rollout_items
}

#[tokio::test]
async fn record_initial_history_reconstructs_typed_inter_agent_message() {
    let (session, _turn_context) = make_session_and_context().await;
    let communication = InterAgentCommunication::new(
        AgentPath::root().join("worker").expect("worker path"),
        AgentPath::root(),
        Vec::new(),
        "child done".to_string(),
        /*trigger_turn*/ false,
    );

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(vec![RolloutItem::InterAgentCommunication(
                communication.clone(),
            )]),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        raw_history_items(&session.state.lock().await.clone_history()),
        vec![communication.to_model_input_item()]
    );
}

#[tokio::test]
async fn record_initial_history_ignores_security_risk_scores() {
    let (session, _turn_context) = make_session_and_context().await;
    let user_item = user_message("visible user input");
    let security_risk = SecurityRiskScore {
        scores: BTreeMap::from([("credential_access".to_string(), 0.92)]),
        call_id: None,
        action: None,
        sampled_at: None,
    };

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(vec![
                RolloutItem::ResponseItem(ResponseItemEnvelope::new(user_item.clone())),
                RolloutItem::SecurityRiskScore(security_risk),
            ]),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        strip_metadata_from_items(&raw_history_items(
            &session.state.lock().await.clone_history()
        )),
        vec![user_item]
    );
}

#[tokio::test]
async fn record_initial_history_restores_world_state_baseline() {
    let (session, turn_context) = make_session_and_context().await;
    let turn_context = Arc::new(turn_context);
    let world_state = build_world_state_from_turn_context(&session, &turn_context).await;
    let expected_history = world_state
        .render_full()
        .into_iter()
        .map(ContextualUserFragment::into_boxed_response_item)
        .collect::<Vec<_>>();
    let mut world_state_items = expected_history
        .iter()
        .cloned()
        .map(ResponseItemEnvelope::new)
        .map(RolloutItem::ResponseItem)
        .collect::<Vec<_>>();
    world_state_items.push(RolloutItem::WorldState(WorldStateItem::full(
        world_state.snapshot().into_object(),
    )));
    let rollout_items =
        completed_user_turn_rollout(turn_context.to_turn_context_item(), world_state_items);

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");
    let step_context = StepContext::for_test(Arc::clone(&turn_context));
    session
        .record_context_updates_and_set_reference_context_item(&step_context)
        .await
        .expect("world state should build");

    assert_eq!(
        raw_history_items(&session.clone_history().await),
        expected_history,
    );
}

#[tokio::test]
async fn record_initial_history_resumed_bare_turn_context_does_not_hydrate_previous_turn_settings()
{
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let rollout_items = vec![RolloutItem::TurnContext(previous_context_item)];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;
    assert_eq!(reconstructed.world_state_baseline, None);

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(session.previous_turn_settings().await, None);
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_hydrates_previous_turn_settings_from_lifecycle_turn_with_missing_turn_context_id()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let mut previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: Some("comp-hash-a".to_string()),
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    previous_context_item.turn_id = None;

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id,
                last_agent_message: None,
                error: None,
                started_at: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: previous_model.to_string(),
            comp_hash: Some("comp-hash-a".to_string()),
            realtime_active: Some(turn_context.realtime_active),
        })
    );
}

#[tokio::test]
async fn reconstruct_history_rollback_keeps_history_and_metadata_in_sync_for_completed_turns() {
    let (session, turn_context) = make_session_and_context().await;
    let first_context_item = turn_context.to_turn_context_item();
    let first_turn_id = first_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let mut rolled_back_context_item = first_context_item.clone();
    rolled_back_context_item.turn_id = Some("rolled-back-turn".to_string());
    rolled_back_context_item.model = "rolled-back-model".to_string();
    let rolled_back_turn_id = rolled_back_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let turn_one_user = user_message("turn 1 user");
    let turn_one_assistant = assistant_message("turn 1 assistant");
    let turn_two_user = user_message("turn 2 user");
    let turn_two_assistant = assistant_message("turn 2 assistant");

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: first_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 1 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(first_context_item.clone()),
        RolloutItem::WorldState(WorldStateItem::full(object!({
            "test": {"environment": "first"}
        }))),
        RolloutItem::ResponseItem(turn_one_user.clone().into()),
        RolloutItem::ResponseItem(turn_one_assistant.clone().into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: first_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: rolled_back_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 2 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(rolled_back_context_item),
        RolloutItem::WorldState(WorldStateItem::patch(object!({
            "test": {"environment": "rolled-back"}
        }))),
        RolloutItem::ResponseItem(turn_two_user.into()),
        RolloutItem::ResponseItem(turn_two_assistant.into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: rolled_back_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        reconstructed.history.as_ref(),
        &annotated(vec![turn_one_user, turn_one_assistant])
    );
    assert_eq!(
        reconstructed.previous_turn_settings,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(reconstructed.reference_context_item)
            .expect("serialize reconstructed reference context item"),
        serde_json::to_value(Some(first_context_item))
            .expect("serialize expected reference context item")
    );
    assert_eq!(
        serde_json::to_value(reconstructed.world_state_baseline)
            .expect("serialize reconstructed world state"),
        json!({"test": {"environment": "first"}})
    );
}

#[tokio::test]
async fn reconstruct_history_rollback_keeps_history_and_metadata_in_sync_for_incomplete_turn() {
    let (session, turn_context) = make_session_and_context().await;
    let first_context_item = turn_context.to_turn_context_item();
    let first_turn_id = first_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let incomplete_turn_id = "incomplete-rolled-back-turn".to_string();
    let turn_one_user = user_message("turn 1 user");
    let turn_one_assistant = assistant_message("turn 1 assistant");
    let turn_two_user = user_message("turn 2 user");

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: first_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 1 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(first_context_item.clone()),
        RolloutItem::ResponseItem(turn_one_user.clone().into()),
        RolloutItem::ResponseItem(turn_one_assistant.clone().into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: first_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: incomplete_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 2 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::ResponseItem(turn_two_user.into()),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        reconstructed.history.as_ref(),
        &annotated(vec![turn_one_user, turn_one_assistant])
    );
    assert_eq!(
        reconstructed.previous_turn_settings,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(reconstructed.reference_context_item)
            .expect("serialize reconstructed reference context item"),
        serde_json::to_value(Some(first_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn reconstruct_history_rollback_skips_non_user_turns_for_history_and_metadata() {
    let (session, turn_context) = make_session_and_context().await;
    let first_context_item = turn_context.to_turn_context_item();
    let first_turn_id = first_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let second_turn_id = "rolled-back-user-turn".to_string();
    let standalone_turn_id = "standalone-turn".to_string();
    let turn_one_user = user_message("turn 1 user");
    let turn_one_assistant = assistant_message("turn 1 assistant");
    let turn_two_user = user_message("turn 2 user");
    let turn_two_assistant = assistant_message("turn 2 assistant");
    let standalone_assistant = assistant_message("standalone assistant");

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: first_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 1 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(first_context_item.clone()),
        RolloutItem::ResponseItem(turn_one_user.clone().into()),
        RolloutItem::ResponseItem(turn_one_assistant.clone().into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: first_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: second_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 2 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::ResponseItem(turn_two_user.into()),
        RolloutItem::ResponseItem(turn_two_assistant.into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: second_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: standalone_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::ResponseItem(standalone_assistant.into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: standalone_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        reconstructed.history.as_ref(),
        &annotated(vec![turn_one_user, turn_one_assistant])
    );
    assert_eq!(
        reconstructed.previous_turn_settings,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(reconstructed.reference_context_item)
            .expect("serialize reconstructed reference context item"),
        serde_json::to_value(Some(first_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn reconstruct_history_rollback_counts_inter_agent_assistant_turns() {
    let (session, turn_context) = make_session_and_context().await;
    let first_context_item = turn_context.to_turn_context_item();
    let first_turn_id = first_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let assistant_turn_id = "assistant-instruction-turn".to_string();
    let assistant_turn_context = TurnContextItem {
        turn_id: Some(assistant_turn_id.clone()),
        ..first_context_item.clone()
    };
    let assistant_instruction = inter_agent_assistant_message("continue");
    let assistant_reply = assistant_message("worker reply");

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: first_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "turn 1 user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(first_context_item.clone()),
        RolloutItem::ResponseItem(user_message("turn 1 user").into()),
        RolloutItem::ResponseItem(assistant_message("turn 1 assistant").into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: first_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: assistant_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::TurnContext(assistant_turn_context),
        RolloutItem::ResponseItem(assistant_instruction.into()),
        RolloutItem::ResponseItem(assistant_reply.into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: assistant_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        reconstructed.history.as_ref(),
        &annotated(vec![
            user_message("turn 1 user"),
            assistant_message("turn 1 assistant")
        ])
    );
    assert_eq!(
        reconstructed.previous_turn_settings,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(reconstructed.reference_context_item)
            .expect("serialize reconstructed reference context item"),
        serde_json::to_value(Some(first_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn reconstruct_history_rollback_clears_history_and_metadata_when_exceeding_user_turns() {
    let (session, turn_context) = make_session_and_context().await;
    let only_context_item = turn_context.to_turn_context_item();
    let only_turn_id = only_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: only_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "only user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(only_context_item),
        RolloutItem::ResponseItem(user_message("only user").into()),
        RolloutItem::ResponseItem(assistant_message("only assistant").into()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: only_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 99 },
        )),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        reconstructed.history.as_ref(),
        &Vec::<ResponseItemEnvelope>::new()
    );
    assert_eq!(reconstructed.previous_turn_settings, None);
    assert!(reconstructed.reference_context_item.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_rollback_skips_only_user_turns() {
    let (session, turn_context) = make_session_and_context().await;
    let previous_context_item = turn_context.to_turn_context_item();
    let user_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let standalone_turn_id = "standalone-task-turn".to_string();
    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: user_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: user_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        // Standalone task turn (no UserMessage) should not consume rollback skips.
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: standalone_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: standalone_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(session.previous_turn_settings().await, None);
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_rollback_drops_incomplete_user_turn_compaction_metadata() {
    let (session, turn_context) = make_session_and_context().await;
    let previous_context_item = turn_context.to_turn_context_item();
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let incomplete_turn_id = "incomplete-compacted-user-turn".to_string();

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item.clone()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: incomplete_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "rolled back".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(session.reference_context_item().await)
            .expect("serialize seeded reference context item"),
        serde_json::to_value(Some(previous_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn record_initial_history_resumed_bare_turn_context_does_not_seed_reference_context_item() {
    let (session, turn_context) = make_session_and_context().await;
    let previous_context_item = turn_context.to_turn_context_item();
    let rollout_items = vec![RolloutItem::TurnContext(previous_context_item.clone())];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_does_not_seed_reference_context_item_after_compaction() {
    let (session, turn_context) = make_session_and_context().await;
    let previous_context_item = turn_context.to_turn_context_item();
    let rollout_items = vec![
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(session.previous_turn_settings().await, None);
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn reconstruct_history_restores_initial_window_from_session_meta() {
    let (session, turn_context) = make_session_and_context().await;
    let thread_id = ThreadId::default();
    let initial_window_id = Uuid::now_v7();
    let rollout_items = vec![RolloutItem::SessionMeta(SessionMetaLine {
        meta: SessionMeta {
            session_id: thread_id.into(),
            id: thread_id,
            context_window: Some(SessionContextWindow {
                window_id: initial_window_id.to_string(),
            }),
            ..SessionMeta::default()
        },
        git: None,
    })];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(reconstructed.window_number, 0);
    assert_eq!(reconstructed.first_window_id, Some(initial_window_id));
    assert_eq!(reconstructed.previous_window_id, None);
    assert_eq!(reconstructed.window_id, Some(initial_window_id));
}

#[tokio::test]
async fn reconstruct_history_prefers_compacted_window_over_session_meta() {
    let (session, turn_context) = make_session_and_context().await;
    let thread_id = ThreadId::default();
    let initial_window_id = Uuid::now_v7();
    let compacted_first_window_id = Uuid::now_v7();
    let compacted_previous_window_id = Uuid::now_v7();
    let compacted_window_id = Uuid::now_v7();
    let rollout_items = vec![
        RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                session_id: thread_id.into(),
                id: thread_id,
                context_window: Some(SessionContextWindow {
                    window_id: initial_window_id.to_string(),
                }),
                ..SessionMeta::default()
            },
            git: None,
        }),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: Some(2),
            first_window_id: Some(compacted_first_window_id.to_string()),
            previous_window_id: Some(compacted_previous_window_id.to_string()),
            window_id: Some(compacted_window_id.to_string()),
        }),
    ];

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(reconstructed.window_number, 2);
    assert_eq!(
        reconstructed.first_window_id,
        Some(compacted_first_window_id)
    );
    assert_eq!(
        reconstructed.previous_window_id,
        Some(compacted_previous_window_id)
    );
    assert_eq!(reconstructed.window_id, Some(compacted_window_id));
}

#[tokio::test]
async fn reconstruct_history_replays_world_state_from_latest_compaction_window() {
    let (session, turn_context) = make_session_and_context().await;
    let rollout_items = completed_user_turn_rollout(
        turn_context.to_turn_context_item(),
        vec![
            RolloutItem::WorldState(WorldStateItem::full(object!({
                "environment": {"status": "old"}
            }))),
            RolloutItem::Compacted(CompactedItem {
                message: String::new(),
                replacement_history: Some(Vec::new()),
                mcp_resource_origins: None,
                window_number: Some(1),
                first_window_id: None,
                previous_window_id: None,
                window_id: None,
            }),
            RolloutItem::WorldState(WorldStateItem::full(object!({
                "environment": {"status": "starting", "cwd": "/workspace"}
            }))),
            RolloutItem::WorldState(WorldStateItem::patch(object!({
                "environment": {"status": "ready"}
            }))),
        ],
    );

    let reconstructed = session
        .reconstruct_history_from_rollout(&turn_context, &rollout_items)
        .await;

    assert_eq!(
        serde_json::to_value(reconstructed.world_state_baseline)
            .expect("serialize reconstructed world state"),
        json!({
            "environment": {"status": "ready", "cwd": "/workspace"}
        })
    );
}

#[tokio::test]
async fn reconstruct_history_rejects_legacy_compaction_without_replacement_history() {
    let (session, turn_context) = make_session_and_context().await;
    let thread_id = ThreadId::default();
    let initial_window_id = Uuid::now_v7();
    let rollout_items = vec![
        RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                session_id: thread_id.into(),
                id: thread_id,
                context_window: Some(SessionContextWindow {
                    window_id: initial_window_id.to_string(),
                }),
                ..SessionMeta::default()
            },
            git: None,
        }),
        RolloutItem::Compacted(CompactedItem {
            message: "legacy summary".to_string(),
            replacement_history: None,
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
    ];

    let error = session
        .reconstruct_resume_state(&turn_context, &rollout_items, None)
        .await
        .expect_err("legacy compaction must be loud");

    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
}

#[tokio::test]
async fn record_initial_history_resumed_turn_context_after_compaction_reestablishes_reference_context_item()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        // Compaction clears baseline until a later TurnContextItem re-establishes it.
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: previous_model.to_string(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(session.reference_context_item().await)
            .expect("serialize seeded reference context item"),
        serde_json::to_value(Some(TurnContextItem {
            turn_id: Some(turn_context.sub_id.clone()),
            #[allow(deprecated)]
            cwd: turn_context.cwd.clone(),
            workspace_roots: None,
            current_date: turn_context.current_date.clone(),
            timezone: turn_context.timezone.clone(),
            approval_policy: turn_context.approval_policy(),
            approvals_reviewer: None,
            sandbox_policy: turn_context.sandbox_policy(),
            permission_profile: None,
            active_permission_profile: None,
            network: None,
            file_system_sandbox_policy: None,
            model: previous_model.to_string(),
            comp_hash: None,
            personality: turn_context.personality(),
            collaboration_mode: Some(turn_context.collaboration_mode()),
            multi_agent_version: None,
            multi_agent_mode: None,
            realtime_active: Some(turn_context.realtime_active),
            cyber_access_program: None,
            effort: turn_context.reasoning_effort().cloned(),
            summary: codex_protocol::config_types::ReasoningSummary::Auto,
        }))
        .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn record_initial_history_resumed_aborted_turn_without_id_clears_active_turn_for_compaction_accounting()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let aborted_turn_id = "aborted-turn-without-id".to_string();

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: aborted_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "aborted".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnAborted(
            codex_protocol::protocol::TurnAbortedEvent {
                turn_id: None,
                started_at: None,
                reason: TurnAbortReason::Interrupted,
                completed_at: None,
                duration_ms: None,
            },
        )),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: previous_model.to_string(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_unmatched_abort_preserves_active_turn_for_later_turn_context()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_context_item = turn_context.to_turn_context_item();
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let current_model = "current-rollout-model";
    let current_turn_id = "current-turn".to_string();
    let unmatched_abort_turn_id = "other-turn".to_string();
    let current_context_item = TurnContextItem {
        turn_id: Some(current_turn_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: current_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: current_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "current".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnAborted(
            codex_protocol::protocol::TurnAbortedEvent {
                turn_id: Some(unmatched_abort_turn_id),
                started_at: None,
                reason: TurnAbortReason::Interrupted,
                completed_at: None,
                duration_ms: None,
            },
        )),
        RolloutItem::TurnContext(current_context_item.clone()),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: current_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: current_model.to_string(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(session.reference_context_item().await)
            .expect("serialize seeded reference context item"),
        serde_json::to_value(Some(current_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn record_initial_history_resumed_trailing_incomplete_turn_compaction_clears_reference_context_item()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let incomplete_turn_id = "trailing-incomplete-turn".to_string();

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: incomplete_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "incomplete".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: previous_model.to_string(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn record_initial_history_resumed_trailing_incomplete_turn_preserves_turn_context_item() {
    let (session, turn_context) = make_session_and_context().await;
    let current_context_item = turn_context.to_turn_context_item();
    let current_turn_id = current_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: current_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "incomplete".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(current_context_item.clone()),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: turn_context.model_info().slug.clone(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert_eq!(
        serde_json::to_value(session.reference_context_item().await)
            .expect("serialize seeded reference context item"),
        serde_json::to_value(Some(current_context_item))
            .expect("serialize expected reference context item")
    );
}

#[tokio::test]
async fn record_initial_history_resumed_replaced_incomplete_compacted_turn_clears_reference_context_item()
 {
    let (session, turn_context) = make_session_and_context().await;
    let previous_model = "previous-rollout-model";
    let previous_context_item = TurnContextItem {
        turn_id: Some(turn_context.sub_id.clone()),
        #[allow(deprecated)]
        cwd: turn_context.cwd.clone(),
        workspace_roots: None,
        current_date: turn_context.current_date.clone(),
        timezone: turn_context.timezone.clone(),
        approval_policy: turn_context.approval_policy(),
        approvals_reviewer: None,
        sandbox_policy: turn_context.sandbox_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: previous_model.to_string(),
        comp_hash: None,
        personality: turn_context.personality(),
        collaboration_mode: Some(turn_context.collaboration_mode()),
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: Some(turn_context.realtime_active),
        cyber_access_program: None,
        effort: turn_context.reasoning_effort().cloned(),
        summary: codex_protocol::config_types::ReasoningSummary::Auto,
    };
    let previous_turn_id = previous_context_item
        .turn_id
        .clone()
        .expect("turn context should have turn_id");
    let compacted_incomplete_turn_id = "compacted-incomplete-turn".to_string();
    let replacing_turn_id = "replacing-turn".to_string();

    let rollout_items = vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: previous_turn_id.clone(),
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "seed".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::TurnContext(previous_context_item),
        RolloutItem::EventMsg(EventMsg::TurnComplete(
            codex_protocol::protocol::TurnCompleteEvent {
                turn_id: previous_turn_id,
                started_at: None,
                last_agent_message: None,
                error: None,
                completed_at: None,
                duration_ms: None,
                time_to_first_token_ms: None,
            },
        )),
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: compacted_incomplete_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "compacted".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::Compacted(CompactedItem {
            message: String::new(),
            replacement_history: Some(Vec::new()),
            mcp_resource_origins: None,
            window_number: None,
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
        // A newer TurnStarted replaces the incomplete compacted turn without a matching
        // completion/abort for the old one.
        RolloutItem::EventMsg(EventMsg::TurnStarted(
            codex_protocol::protocol::TurnStartedEvent {
                turn_id: replacing_turn_id,
                trace_id: None,
                started_at: None,
                model_context_window: Some(128_000),
                collaboration_mode_kind: ModeKind::Default,
            },
        )),
    ];

    session
        .record_initial_history(InitialHistory::Resumed(ResumedHistory {
            conversation_id: ThreadId::default(),
            history: Arc::new(rollout_items),
            rollout_path: Some(PathBuf::from("/tmp/resume.jsonl")),
            materialized_resume: None,
        }))
        .await
        .expect("record initial history");

    assert_eq!(
        session.previous_turn_settings().await,
        Some(PreviousTurnSettings {
            model: previous_model.to_string(),
            comp_hash: None,
            realtime_active: Some(turn_context.realtime_active),
        })
    );
    assert!(session.reference_context_item().await.is_none());
}

#[tokio::test]
async fn checkpoint_suffix_replay_matches_full_replay_after_rollback() {
    let (session, turn_context) = make_session_and_context().await;
    let thread_id = ThreadId::new();
    let first_window_id = Uuid::now_v7();
    let current_window_id = Uuid::now_v7();
    let mut first_context = turn_context.to_turn_context_item();
    first_context.turn_id = Some("turn-1".to_string());
    let mut rolled_back_context = first_context.clone();
    rolled_back_context.turn_id = Some("turn-2".to_string());
    rolled_back_context.model = "rolled-back-model".to_string();
    let mut suffix_context = first_context.clone();
    suffix_context.turn_id = Some("turn-3".to_string());
    suffix_context.model = "suffix-model".to_string();
    let session_meta = RolloutItem::SessionMeta(SessionMetaLine {
        meta: SessionMeta {
            session_id: thread_id.into(),
            id: thread_id,
            context_window: Some(SessionContextWindow::new(first_window_id.to_string())),
            ..SessionMeta::default()
        },
        git: None,
    });
    let first_history = annotated(vec![
        user_message("first user"),
        assistant_message("first assistant"),
    ]);
    let prefix = vec![
        session_meta.clone(),
        turn_started("turn-1"),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "first user".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::ResponseItem(user_message("first user").into()),
        RolloutItem::ResponseItem(assistant_message("first assistant").into()),
        RolloutItem::TurnContext(first_context.clone()),
        RolloutItem::Compacted(CompactedItem {
            message: "summary".to_string(),
            replacement_history: Some(first_history),
            mcp_resource_origins: None,
            window_number: Some(3),
            first_window_id: Some(first_window_id.to_string()),
            previous_window_id: Some(first_window_id.to_string()),
            window_id: Some(current_window_id.to_string()),
        }),
        RolloutItem::WorldState(WorldStateItem::full(object!({
            "repository": {"head": "first"}
        }))),
        turn_complete("turn-1"),
        turn_started("turn-2"),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "rolled back".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::ResponseItem(user_message("rolled back").into()),
        RolloutItem::TurnContext(rolled_back_context),
        RolloutItem::WorldState(WorldStateItem::patch(object!({
            "repository": {"head": "rolled-back"}
        }))),
        turn_complete("turn-2"),
        RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
            codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
        )),
    ];
    let suffix = vec![
        turn_started("turn-3"),
        RolloutItem::EventMsg(EventMsg::UserMessage(
            codex_protocol::protocol::UserMessageEvent {
                client_id: None,
                message: "suffix".to_string(),
                images: None,
                local_images: Vec::new(),
                text_elements: Vec::new(),
                ..Default::default()
            },
        )),
        RolloutItem::ResponseItem(user_message("suffix").into()),
        RolloutItem::TurnContext(suffix_context),
        RolloutItem::WorldState(WorldStateItem::patch(object!({
            "repository": {"head": "suffix"}
        }))),
        turn_complete("turn-3"),
    ];

    let prefix_state = session
        .reconstruct_resume_state(&turn_context, &prefix, None)
        .await
        .expect("reconstruct checkpoint prefix");
    let materialized_state = MaterializedResumeState {
        version: MATERIALIZED_RESUME_STATE_VERSION,
        history: Arc::clone(&prefix_state.history),
        previous_turn_settings: prefix_state
            .previous_turn_settings
            .as_ref()
            .map(|settings| MaterializedPreviousTurnSettings {
                model: settings.model.clone(),
                comp_hash: settings.comp_hash.clone(),
                realtime_active: settings.realtime_active,
            }),
        reference_context_item: prefix_state.reference_context_item.clone(),
        world_state_baseline: prefix_state
            .world_state_baseline
            .clone()
            .map(|snapshot| WorldStateItem::full(snapshot.into_object())),
        mcp_resource_origins: prefix_state.mcp_resource_origins.clone(),
        auto_compact_window: MaterializedAutoCompactWindow {
            window_number: prefix_state.window_number,
            first_window_id: prefix_state
                .first_window_id
                .expect("first window ID")
                .to_string(),
            previous_window_id: prefix_state.previous_window_id.map(|id| id.to_string()),
            window_id: prefix_state.window_id.expect("window ID").to_string(),
        },
        token_info: prefix_state.token_info.clone(),
        last_agent_status: prefix_state.last_agent_status.clone(),
        truncation_policy: TruncationPolicy::from(turn_context.model_info().truncation_policy),
        estimated_prefill_input_tokens: None,
    };
    let mut full_items = prefix;
    full_items.extend(suffix.clone());
    let full = session
        .reconstruct_resume_state(&turn_context, &full_items, None)
        .await
        .expect("full reconstruction");
    let suffix_items = std::iter::once(session_meta)
        .chain(suffix)
        .collect::<Vec<_>>();
    let checkpoint_suffix = session
        .reconstruct_resume_state(&turn_context, &suffix_items, Some(&materialized_state))
        .await
        .expect("checkpoint suffix reconstruction");

    assert_eq!(checkpoint_suffix, full);
    assert_eq!(checkpoint_suffix.window_number, 3);
    assert_eq!(
        checkpoint_suffix
            .reference_context_item
            .as_ref()
            .map(|context| context.model.as_str()),
        Some("suffix-model")
    );
    assert_eq!(
        serde_json::to_value(
            checkpoint_suffix
                .world_state_baseline
                .clone()
                .expect("world state")
                .into_object()
        )
        .expect("serialize world state"),
        json!({"repository": {"head": "suffix"}})
    );

    let error = session
        .reconstruct_resume_state(
            &turn_context,
            &[RolloutItem::EventMsg(EventMsg::ThreadRolledBack(
                codex_protocol::protocol::ThreadRolledBackEvent { num_turns: 1 },
            ))],
            Some(&materialized_state),
        )
        .await
        .expect_err("rollback across fence must fail");
    assert!(
        error
            .to_string()
            .contains("codex_resume_state_needs_compaction"),
        "{error}"
    );
}
