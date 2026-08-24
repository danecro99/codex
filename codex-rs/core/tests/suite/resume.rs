use anyhow::Result;
use codex_core::TurnInputRequest;
use codex_history::CompactedItem;
use codex_history::RolloutItem;
use codex_history::RolloutLine;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::SandboxPolicy;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::ThreadHistoryMode;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::protocol::TurnCompleteEvent;
use codex_protocol::protocol::TurnContextItem;
use codex_protocol::protocol::TurnStartedEvent;
use codex_protocol::protocol::UserMessageEvent;
use codex_protocol::user_input::ByteRange;
use codex_protocol::user_input::TextElement;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_reasoning_item;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use core_test_support::wait_for_event;
use pretty_assertions::assert_eq;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use tempfile::TempDir;

fn write_legacy_rollout(
    path: &Path,
    session_meta: SessionMetaLine,
    items: Vec<RolloutItem>,
) -> Result<()> {
    std::fs::create_dir_all(path.parent().expect("rollout path should have a parent"))?;
    let mut file = std::fs::File::create(path)?;
    for item in std::iter::once(RolloutItem::SessionMeta(session_meta)).chain(items) {
        writeln!(
            file,
            "{}",
            serde_json::to_string(&RolloutLine {
                timestamp: "2026-08-25T00:00:00Z".to_string(),
                ordinal: None,
                item,
            })?
        )?;
    }
    Ok(())
}

fn user_response(text: &str) -> ResponseItem {
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

fn turn_context(cwd: &Path, turn_id: &str, model: &str) -> RolloutItem {
    RolloutItem::TurnContext(TurnContextItem {
        turn_id: Some(turn_id.to_string()),
        cwd: serde_json::from_value(serde_json::json!(cwd)).expect("absolute cwd"),
        workspace_roots: None,
        current_date: Some("2026-08-24".to_string()),
        timezone: Some("Europe/Berlin".to_string()),
        approval_policy: AskForApproval::Never,
        approvals_reviewer: None,
        sandbox_policy: SandboxPolicy::new_read_only_policy(),
        permission_profile: None,
        active_permission_profile: None,
        network: None,
        file_system_sandbox_policy: None,
        model: model.to_string(),
        comp_hash: None,
        personality: None,
        collaboration_mode: None,
        multi_agent_version: None,
        multi_agent_mode: None,
        realtime_active: None,
        effort: None,
        summary: ReasoningSummary::Auto,
    })
}

fn legacy_session_meta(cwd: &Path, thread_id: ThreadId, timestamp: &str) -> SessionMetaLine {
    SessionMetaLine {
        meta: SessionMeta {
            session_id: thread_id.into(),
            id: thread_id,
            timestamp: timestamp.to_string(),
            cwd: cwd.to_path_buf(),
            originator: "resume-integration-test".to_string(),
            cli_version: "0.149.1-test".to_string(),
            model_provider: Some("openai".to_string()),
            history_mode: ThreadHistoryMode::Legacy,
            ..Default::default()
        },
        git: None,
    }
}

fn compacted_legacy_turn(
    cwd: &Path,
    turn_id: &str,
    replacement: &str,
    model: &str,
) -> Vec<RolloutItem> {
    vec![
        RolloutItem::EventMsg(EventMsg::TurnStarted(TurnStartedEvent {
            turn_id: turn_id.to_string(),
            trace_id: None,
            started_at: None,
            model_context_window: Some(128_000),
            collaboration_mode_kind: Default::default(),
        })),
        RolloutItem::EventMsg(EventMsg::UserMessage(UserMessageEvent {
            message: format!("{turn_id} before compaction"),
            ..Default::default()
        })),
        RolloutItem::Compacted(CompactedItem {
            message: format!("{turn_id} checkpoint"),
            replacement_history: Some(vec![user_response(replacement).into()]),
            mcp_resource_origins: None,
            window_number: Some(1),
            first_window_id: None,
            previous_window_id: None,
            window_id: None,
        }),
        turn_context(cwd, turn_id, model),
        RolloutItem::EventMsg(EventMsg::TurnComplete(TurnCompleteEvent {
            turn_id: turn_id.to_string(),
            last_agent_message: None,
            error: None,
            started_at: None,
            completed_at: None,
            duration_ms: None,
            time_to_first_token_ms: None,
        })),
    ]
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_includes_initial_messages_from_rollout_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_assistant_message("msg-1", "Completed first turn"),
        ev_completed("resp-initial"),
    ]);
    mount_sse_once(&server, initial_sse).await;

    let text_elements = vec![TextElement::new(
        ByteRange { start: 0, end: 6 },
        Some("<note>".into()),
    )];

    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Record some messages".into(),
            text_elements: text_elements.clone(),
        }]))
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let resumed = builder.restart(&server, &initial).await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
        ] => {
            assert_eq!(first_user.message, "Record some messages");
            assert_eq!(first_user.text_elements, text_elements);
            assert_eq!(assistant_message.message, "Completed first turn");
            assert_eq!(completed.turn_id, started.turn_id);
            assert_eq!(
                completed.last_agent_message.as_deref(),
                Some("Completed first turn")
            );
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_includes_initial_messages_from_reasoning_events() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.show_raw_agent_reasoning = true;
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_reasoning_item("reason-1", &["Summarized step"], &["raw detail"]),
        ev_assistant_message("msg-1", "Completed reasoning turn"),
        ev_completed("resp-initial"),
    ]);
    mount_sse_once(&server, initial_sse).await;

    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Record reasoning messages".into(),
            text_elements: Vec::new(),
        }]))
        .await?;

    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let resumed = builder.restart(&server, &initial).await?;
    let initial_messages = resumed
        .session_configured
        .initial_messages
        .expect("expected initial messages to be present for resumed session");
    match initial_messages.as_slice() {
        [
            EventMsg::TurnStarted(started),
            EventMsg::UserMessage(first_user),
            EventMsg::AgentReasoning(reasoning),
            EventMsg::AgentReasoningRawContent(raw),
            EventMsg::AgentMessage(assistant_message),
            EventMsg::TokenCount(_),
            EventMsg::TurnComplete(completed),
        ] => {
            assert_eq!(first_user.message, "Record reasoning messages");
            assert_eq!(reasoning.text, "Summarized step");
            assert_eq!(raw.text, "raw detail");
            assert_eq!(assistant_message.message, "Completed reasoning turn");
            assert_eq!(completed.turn_id, started.turn_id);
            assert_eq!(
                completed.last_agent_message.as_deref(),
                Some("Completed reasoning turn")
            );
        }
        other => panic!("unexpected initial messages after resume: {other:#?}"),
    }

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_switches_models_preserves_base_instructions() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2".to_string());
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let initial_sse = sse(vec![
        ev_response_created("resp-initial"),
        ev_assistant_message("msg-1", "Completed first turn"),
        ev_completed("resp-initial"),
    ]);
    let initial_mock = mount_sse_once(&server, initial_sse).await;

    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Record initial instructions".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let initial_body = initial_mock.single_request().body_json();
    let initial_instructions = initial_body
        .get("instructions")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();

    let resumed_mock = mount_sse_sequence(
        &server,
        vec![
            sse(vec![
                ev_response_created("resp-resume-1"),
                ev_assistant_message("msg-2", "Resumed turn"),
                ev_completed("resp-resume-1"),
            ]),
            sse(vec![
                ev_response_created("resp-resume-2"),
                ev_assistant_message("msg-3", "Second resumed turn"),
                ev_completed("resp-resume-2"),
            ]),
        ],
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.4".to_string());
    });
    let resumed = resume_builder.restart(&server, &initial).await?;
    resumed
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Resume with different model".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    resumed
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Second turn after resume".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let requests = resumed_mock.requests();
    assert_eq!(requests.len(), 2, "expected two resumed requests");

    let first_resumed = &requests[0];
    assert_eq!(first_resumed.instructions_text(), initial_instructions);
    let first_developer_texts = first_resumed.message_input_texts("developer");
    let first_model_switch_count = first_developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert!(
        first_model_switch_count >= 1,
        "expected model switch message on first post-resume turn"
    );

    let second_resumed = &requests[1];
    assert_eq!(second_resumed.instructions_text(), initial_instructions);
    let second_developer_texts = second_resumed.message_input_texts("developer");
    let second_model_switch_count = second_developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert_eq!(
        second_model_switch_count, 1,
        "did not expect duplicate model switch message after first post-resume turn"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resume_model_switch_is_not_duplicated_after_pre_turn_override() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.2".to_string());
    });
    let initial = builder.build(&server).await?;
    let codex = Arc::clone(&initial.codex);

    let initial_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-initial"),
            ev_assistant_message("msg-1", "Completed first turn"),
            ev_completed("resp-initial"),
        ]),
    )
    .await;
    codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "Record initial instructions".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&codex, |event| matches!(event, EventMsg::TurnComplete(_))).await;
    let _ = initial_mock.single_request();

    let resumed_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-resume"),
            ev_assistant_message("msg-2", "Resumed turn"),
            ev_completed("resp-resume"),
        ]),
    )
    .await;

    let mut resume_builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.5".to_string());
    });
    let resumed = resume_builder.restart(&server, &initial).await?;
    core_test_support::submit_thread_settings(
        &resumed.codex,
        ThreadSettingsOverrides {
            model: Some("gpt-5.4".to_string()),
            ..Default::default()
        },
    )
    .await?;
    resumed
        .codex
        .start_or_steer_turn(TurnInputRequest::user_input(vec![UserInput::Text {
            text: "first turn after override".into(),
            text_elements: Vec::new(),
        }]))
        .await?;
    wait_for_event(&resumed.codex, |event| {
        matches!(event, EventMsg::TurnComplete(_))
    })
    .await;

    let request = resumed_mock.single_request();
    let developer_texts = request.message_input_texts("developer");
    let model_switch_count = developer_texts
        .iter()
        .filter(|text| text.contains("<model_switch>"))
        .count();
    assert_eq!(model_switch_count, 1);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn path_resume_sends_bounded_compacted_model_context() -> Result<()> {
    skip_if_no_network!(Ok(()));

    const EXCLUDED_OLD_PREFIX: &str = "excluded old prefix";
    const SELECTED_REPLACEMENT: &str = "selected replacement history";
    const OTHER_REPLACEMENT: &str = "other rollout replacement history";
    const RESUMED_USER_MESSAGE: &str = "continue the selected rollout";

    let thread_id = ThreadId::from_string("018f0000-0000-7000-8000-000000000001")?;
    let home = Arc::new(TempDir::new()?);
    let selected_path = home.path().join(format!(
        "sessions/2026/08/25/rollout-2026-08-25T12-00-00-{thread_id}.jsonl"
    ));
    let other_path = home.path().join(format!(
        "sessions/2026/08/25/rollout-2026-08-25T12-01-00-{thread_id}.jsonl"
    ));
    let selected_meta = legacy_session_meta(home.path(), thread_id, "2026-08-25T12:00:00Z");
    let other_meta = legacy_session_meta(home.path(), thread_id, "2026-08-25T12:01:00Z");

    let mut selected_items = vec![RolloutItem::ResponseItem(
        user_response(EXCLUDED_OLD_PREFIX).into(),
    )];
    selected_items.extend(compacted_legacy_turn(
        home.path(),
        "selected-turn",
        SELECTED_REPLACEMENT,
        "gpt-5.2",
    ));
    write_legacy_rollout(selected_path.as_path(), selected_meta, selected_items)?;
    write_legacy_rollout(
        other_path.as_path(),
        other_meta,
        compacted_legacy_turn(home.path(), "other-turn", OTHER_REPLACEMENT, "gpt-5.4"),
    )?;

    let server = start_mock_server().await;
    let response_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-resume-selected"),
            ev_completed("resp-resume-selected"),
        ]),
    )
    .await;
    let mut builder = test_codex().with_config(|config| {
        config.model = Some("gpt-5.4".to_string());
    });
    let resumed = builder
        .resume(&server, Arc::clone(&home), selected_path.clone())
        .await?;
    resumed.submit_turn(RESUMED_USER_MESSAGE).await?;

    let request = response_mock.single_request();
    let user_texts = request.message_input_texts("user");
    let replacement_index = user_texts
        .iter()
        .position(|text| text == SELECTED_REPLACEMENT)
        .expect("selected replacement history should reach the model request");
    let resumed_message_index = user_texts
        .iter()
        .position(|text| text == RESUMED_USER_MESSAGE)
        .expect("resumed user message should reach the model request");
    assert!(replacement_index < resumed_message_index);
    assert!(!request.body_contains_text(EXCLUDED_OLD_PREFIX));
    assert!(!request.body_contains_text(OTHER_REPLACEMENT));
    assert_eq!(
        request
            .message_input_texts("developer")
            .iter()
            .filter(|text| text.contains("<model_switch>"))
            .count(),
        1,
        "the selected rollout's gpt-5.2 TurnContext should reach the gpt-5.4 resumed turn"
    );

    Ok(())
}
