use std::path::Path;
use std::path::PathBuf;

use codex_protocol::ResponseItemId;
use codex_protocol::ThreadId;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ContentItemKind;
use codex_protocol::models::ExecutedToolCall;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::InternalChatMessageMetadataPassthrough;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningItemReasoningSummary;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionMeta;
use codex_protocol::protocol::SessionMetaLine;
use codex_protocol::protocol::ThreadHistoryMode;
use pretty_assertions::assert_eq;
use serde_json::Value;
use tempfile::TempDir;

use crate::RolloutItem;
use crate::RolloutLine;
use crate::append_rollout_item_to_path;
use crate::decode_rollout_line;
use crate::persisted_rollout_item;

fn response_item(item: ResponseItem) -> RolloutItem {
    RolloutItem::ResponseItem(item.into())
}

fn reasoning(content: Option<Vec<ReasoningItemContent>>) -> RolloutItem {
    response_item(ResponseItem::Reasoning {
        id: Some(ResponseItemId::with_suffix("rs", "1")),
        summary: vec![ReasoningItemReasoningSummary::SummaryText {
            text: "summary".to_string(),
        }],
        content,
        encrypted_content: Some("encrypted".to_string()),
        internal_chat_message_metadata_passthrough: None,
    })
}

fn message(role: &str, content: ContentItem) -> RolloutItem {
    response_item(ResponseItem::Message {
        id: None,
        role: role.to_string(),
        content: vec![content],
        phase: None,
        internal_chat_message_metadata_passthrough: None,
    })
}

/// Passthrough metadata carrying every field the persisted encoding treats asymmetrically.
///
/// `cell_id`, `executed_tool_calls` and `tool_calls_complete` serialize normally but are
/// `skip_deserializing`, so a record carries them to disk and can never read them back.
/// `create_time` is a `serde_json::Number`, the nested-number shape that `decode_rollout_line`
/// exists to keep decodable under `serde_json/arbitrary_precision`.
fn lossy_metadata() -> InternalChatMessageMetadataPassthrough {
    InternalChatMessageMetadataPassthrough {
        turn_id: Some("turn-1".to_string()),
        create_time: Some(
            serde_json::Number::from_f64(1_757_000_000.125).expect("finite create time"),
        ),
        content_item_kinds: Some(vec![ContentItemKind("text".to_string())]),
        cell_id: Some("cell-1".to_string()),
        executed_tool_calls: Some(vec![ExecutedToolCall::new(
            "shell".to_string(),
            serde_json::json!({"command": ["echo", "hi"]}),
        )]),
        tool_calls_complete: Some(true),
    }
}

fn message_with_lossy_metadata() -> RolloutItem {
    response_item(ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: "carrying host metadata".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: Some(lossy_metadata()),
    })
}

/// A tool call whose arguments are a nested `serde_json::Value` with deliberately unsorted keys
/// and a fractional number, so map ordering and number formatting are both exercised.
fn tool_search_call_with_nested_value() -> RolloutItem {
    response_item(ResponseItem::ToolSearchCall {
        id: None,
        call_id: Some("call-2".to_string()),
        status: None,
        execution: "completed".to_string(),
        arguments: serde_json::json!({
            "zeta": 1,
            "alpha": {"nested_zeta": true, "nested_alpha": [1, 2, 3]},
            "ratio": 0.125,
        }),
        internal_chat_message_metadata_passthrough: Some(lossy_metadata()),
    })
}

fn function_call() -> RolloutItem {
    response_item(ResponseItem::FunctionCall {
        id: None,
        name: "shell".to_string(),
        namespace: None,
        arguments: r#"{"command":["echo","hi"]}"#.to_string(),
        encrypted_function_args: None,
        call_id: "call-1".to_string(),
        internal_chat_message_metadata_passthrough: None,
    })
}

fn function_call_output() -> RolloutItem {
    response_item(ResponseItem::FunctionCallOutput {
        id: None,
        call_id: Some("call-1".to_string()),
        name: None,
        namespace: None,
        output: FunctionCallOutputPayload {
            body: FunctionCallOutputBody::Text("hi\n".to_string()),
            success: Some(true),
        },
        internal_chat_message_metadata_passthrough: None,
    })
}

fn turn_items() -> Vec<RolloutItem> {
    vec![
        message(
            "user",
            ContentItem::InputText {
                text: "hello".to_string(),
            },
        ),
        reasoning(Some(vec![ReasoningItemContent::Text {
            text: "raw reasoning".to_string(),
        }])),
        reasoning(Some(Vec::new())),
        reasoning(Some(vec![ReasoningItemContent::ReasoningText {
            text: "visible reasoning".to_string(),
        }])),
        reasoning(None),
        function_call(),
        function_call_output(),
        message_with_lossy_metadata(),
        tool_search_call_with_nested_value(),
        message(
            "assistant",
            ContentItem::OutputText {
                text: "done".to_string(),
            },
        ),
    ]
}

fn seed_paginated_rollout(home: &Path) -> PathBuf {
    let thread_id = ThreadId::new();
    let path = home.join(format!("rollout-2026-09-06T00-00-00-{thread_id}.jsonl"));
    let session_meta = RolloutLine {
        timestamp: "2026-09-06T00:00:00.000Z".to_string(),
        ordinal: Some(0),
        item: RolloutItem::SessionMeta(SessionMetaLine {
            meta: SessionMeta {
                session_id: thread_id.into(),
                id: thread_id,
                history_mode: ThreadHistoryMode::Paginated,
                ..SessionMeta::default()
            },
            git: None,
        }),
    };
    let mut encoded = serde_json::to_string(&session_meta).expect("encode session meta");
    encoded.push('\n');
    std::fs::write(path.as_path(), encoded).expect("seed rollout");
    path
}

fn json(item: &RolloutItem) -> Value {
    serde_json::to_value(item).expect("serialize rollout item")
}

/// The exact bytes `append_generation::hash_item` fingerprints.
///
/// `Value` comparison is not enough here: with `serde_json/preserve_order` two objects that differ
/// only in key order compare equal while their serialized bytes differ, and the durability
/// fingerprint is taken over the bytes.
fn fingerprint_bytes(item: &RolloutItem) -> Vec<u8> {
    serde_json::to_vec(item).expect("fingerprint rollout item")
}

/// Reports which `serde_json` features this build unified, so a green run states what it covered.
fn serde_json_features() -> String {
    let arbitrary_precision = serde_json::to_string(
        &serde_json::from_str::<Value>("1.000").expect("parse fixed-point number"),
    )
    .expect("re-serialize number")
        == "1.000";
    let preserve_order = serde_json::to_string(
        &serde_json::from_str::<Value>(r#"{"b":1,"a":2}"#).expect("parse unsorted object"),
    )
    .expect("re-serialize object")
        == r#"{"b":1,"a":2}"#;
    format!("arbitrary_precision={arbitrary_precision} preserve_order={preserve_order}")
}

fn payload_field<'a>(item: &'a Value, field: &str) -> Option<&'a Value> {
    item.get("payload")?.get(field)
}

fn payload_metadata(item: &Value) -> Value {
    payload_field(item, "internal_chat_message_metadata_passthrough")
        .cloned()
        .unwrap_or(Value::Null)
}

/// The persisted encoding is deliberately lossy, so a record can never reproduce every in-memory
/// field. A producer that fingerprints a write intent must therefore describe the persisted form.
#[tokio::test]
async fn durable_records_decode_to_the_persisted_form_of_every_turn_item() {
    let home = TempDir::new().expect("temp dir");
    let path = seed_paginated_rollout(home.path());
    let items = turn_items();
    for item in &items {
        append_rollout_item_to_path(path.as_path(), item)
            .await
            .expect("append rollout item");
    }

    let transcript = std::fs::read_to_string(path.as_path()).expect("read rollout");
    let decoded = transcript
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            decode_rollout_line(serde_json::from_str(line).expect("record is valid json"))
                .expect("record decodes")
        })
        .collect::<Vec<_>>();

    assert_eq!(decoded.len(), items.len() + 1);
    assert_eq!(
        decoded.iter().map(|line| line.ordinal).collect::<Vec<_>>(),
        (0..=u64::try_from(items.len()).expect("ordinal"))
            .map(Some)
            .collect::<Vec<_>>()
    );
    // This is the equality `append_generation` depends on: the fingerprint of the write intent,
    // taken over the persisted form, against the fingerprint of what the durable record decodes to.
    for (item, line) in items.iter().zip(decoded.iter().skip(1)) {
        let intent = persisted_rollout_item(item).expect("normalize written item");
        assert_eq!(
            fingerprint_bytes(&line.item),
            fingerprint_bytes(&intent),
            "durable record and write intent disagree for {item:?} with {}",
            serde_json_features()
        );
    }
}

/// Reasoning whose content carries no `reasoning_text` is dropped on the way to disk and decodes
/// back as `None`, which serializes again as an explicit `null`. Fingerprinting the raw item would
/// therefore reject the record that faithfully carried it.
#[test]
fn reasoning_without_reasoning_text_is_not_a_serialization_fixed_point() {
    for content in [
        Some(vec![ReasoningItemContent::Text {
            text: "raw reasoning".to_string(),
        }]),
        Some(Vec::new()),
    ] {
        let item = reasoning(content);
        let persisted = persisted_rollout_item(&item).expect("normalize reasoning");
        assert_ne!(json(&item), json(&persisted));
        assert_eq!(payload_field(&json(&item), "content"), None);
        assert_eq!(
            payload_field(&json(&persisted), "content"),
            Some(&Value::Null)
        );
    }
}

/// `ResponseItem::Reasoning::content` is not the only field the persisted encoding drops.
///
/// Every `skip_deserializing` field in `InternalChatMessageMetadataPassthrough` serializes into the
/// durable record and then decodes back as `None`. A write intent taken over the raw in-memory item
/// is therefore unusable for more than one reason, which is why the intent is normalized through
/// the persisted form rather than by fixing individual fields.
#[test]
fn host_owned_passthrough_metadata_is_written_but_never_read_back() {
    let item = message_with_lossy_metadata();
    let persisted = persisted_rollout_item(&item).expect("normalize metadata item");
    let written = payload_metadata(&json(&item));
    let read_back = payload_metadata(&json(&persisted));

    assert_ne!(json(&item), json(&persisted));
    for field in ["cell_id", "executed_tool_calls", "tool_calls_complete"] {
        assert!(written.get(field).is_some(), "{field} must reach the record");
        assert_eq!(
            read_back.get(field),
            None,
            "{field} cannot be read back out of the record"
        );
    }
    // Fields without `skip_deserializing` survive, including the nested number.
    for field in ["turn_id", "create_time", "content_item_kinds"] {
        assert_eq!(written.get(field), read_back.get(field), "{field}");
    }
}

/// Normalizing once is enough: the persisted form is stable under further round trips, so both
/// sides of a durability fingerprint converge on the same bytes.
#[test]
fn the_persisted_form_is_stable_under_further_round_trips() {
    for item in turn_items() {
        let persisted = persisted_rollout_item(&item).expect("normalize item");
        let twice = persisted_rollout_item(&persisted).expect("normalize persisted item");
        assert_eq!(json(&persisted), json(&twice));
    }
}
