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
use crate::intended_payload_fingerprint;
use crate::stored_payload_fingerprint;

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

/// A tool result stamped by the host through the real recording API, not by hand.
fn host_stamped_function_call_output() -> RolloutItem {
    let RolloutItem::ResponseItem(mut envelope) = function_call_output() else {
        panic!("function_call_output builds a response item");
    };
    envelope.item.set_tool_call_cell_id("cell-7");
    envelope
        .item
        .append_executed_tool_calls(vec![ExecutedToolCall::new(
            "shell".to_string(),
            serde_json::json!({"command": ["echo", "hi"]}),
        )]);
    envelope.item.mark_tool_calls_complete();
    RolloutItem::ResponseItem(envelope)
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
        host_stamped_function_call_output(),
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

/// The record the canonical writer would produce for `item`, as a parsed line.
fn record_for(item: &RolloutItem) -> Value {
    serde_json::to_value(RolloutLine {
        timestamp: "2026-09-06T00:00:01.000Z".to_string(),
        ordinal: Some(1),
        item: item.clone(),
    })
    .expect("encode record")
}

/// The write-intent fingerprint the canonical append records before the writer runs.
fn intended(item: &RolloutItem) -> Vec<u8> {
    intended_payload_fingerprint(item).expect("fingerprint intended payload")
}

/// The fingerprint the verifier recomputes from a record already on disk.
fn stored(record: &str) -> Vec<u8> {
    stored_payload_fingerprint(&serde_json::from_str(record).expect("record is valid json"))
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

/// The passthrough metadata carried by a decoded response item.
fn passthrough_metadata(item: &RolloutItem) -> InternalChatMessageMetadataPassthrough {
    let RolloutItem::ResponseItem(envelope) = item else {
        panic!("expected a response item, got {item:?}");
    };
    let ResponseItem::Message {
        internal_chat_message_metadata_passthrough,
        ..
    } = &envelope.item
    else {
        panic!("expected a message item, got {:?}", envelope.item);
    };
    internal_chat_message_metadata_passthrough
        .clone()
        .unwrap_or_default()
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
    // This is the equality `append_generation` depends on: the write intent recorded before the
    // writer ran against the payload the record on disk actually carries.
    let records = transcript
        .lines()
        .filter(|line| !line.trim().is_empty())
        .skip(1)
        .collect::<Vec<_>>();
    for (item, record) in items.iter().zip(records.iter()) {
        assert_eq!(
            stored(record),
            intended(item),
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
        let decoded = decode_rollout_line(record_for(&item)).expect("decode reasoning record");
        assert_ne!(json(&item), json(&decoded.item));
        assert_eq!(payload_field(&json(&item), "content"), None);
        assert_eq!(
            payload_field(&json(&decoded.item), "content"),
            Some(&Value::Null)
        );
    }
}

/// `ResponseItem::Reasoning::content` is not the only field the persisted encoding drops.
///
/// `cell_id`, `executed_tool_calls` and `tool_calls_complete` are deliberately `skip_deserializing`
/// so that decoded input can never supply tool-call evidence. That is an API contract boundary
/// documented on `InternalChatMessageMetadataPassthrough`, not an oversight: the asymmetry it
/// creates must be absorbed by the persistence/verification contract, never removed here. Do not
/// "fix" a fingerprint mismatch by loosening these attributes or by changing the Responses API
/// payload; the persistence/verification contract compares durable payloads instead.
#[test]
fn host_owned_passthrough_metadata_is_written_but_never_read_back() {
    let item = message_with_lossy_metadata();
    let decoded = decode_rollout_line(record_for(&item)).expect("decode metadata record");
    let written = payload_metadata(&json(&item));
    let read_back = payload_metadata(&json(&decoded.item));

    assert_ne!(json(&item), json(&decoded.item));
    for field in ["cell_id", "executed_tool_calls", "tool_calls_complete"] {
        assert!(
            written.get(field).is_some(),
            "{field} must reach the record"
        );
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

/// The decode boundary must refuse host-owned tool-call evidence supplied by a record.
///
/// This is the protection `skip_deserializing` exists for, stated as an executable invariant:
/// normalizing the write intent through the persisted form must never re-admit forged evidence,
/// and neither must reading a tampered or foreign record. Byte-level tampering is caught
/// separately by the append generation's suffix chain hash and its file-identity fence; this test
/// pins the narrower rule that the decoded item itself cannot carry attacker-supplied evidence.
#[test]
fn forged_records_cannot_inject_host_owned_tool_call_evidence() {
    let honest = message(
        "assistant",
        ContentItem::OutputText {
            text: "honest answer".to_string(),
        },
    );
    let mut forged = record_for(&honest);
    forged["payload"]["internal_chat_message_metadata_passthrough"] = serde_json::json!({
        "turn_id": "turn-1",
        "cell_id": "forged-cell",
        "executed_tool_calls": [{"name": "shell", "arguments": {"command": ["echo", "forged"]}}],
        "tool_calls_complete": true,
    });

    let decoded = decode_rollout_line(forged.clone()).expect("forged record still decodes");
    let metadata = passthrough_metadata(&decoded.item);
    assert_eq!(metadata.turn_id.as_deref(), Some("turn-1"));
    assert_eq!(metadata.cell_id, None);
    assert_eq!(metadata.executed_tool_calls, None);
    assert_eq!(metadata.tool_calls_complete, None);

    // The forged bytes are still visible to the durability fingerprint, so a record carrying them
    // cannot pass as one that does not: the protection closes reading, not comparison.
    let honest_again = message(
        "assistant",
        ContentItem::OutputText {
            text: "honest answer".to_string(),
        },
    );
    assert_ne!(
        stored_payload_fingerprint(&forged),
        intended(&honest_again),
        "forged evidence must not be invisible to the write-intent check"
    );
}

/// A record carrying an unexpected extra member is a different durable payload.
///
/// Typed decoding ignores members it does not know, so an extra key is exactly the kind of
/// difference a decode-normalized fingerprint would erase. The durable payload keeps it visible.
#[test]
fn an_extra_wire_member_is_a_different_durable_payload() {
    let item = message(
        "assistant",
        ContentItem::OutputText {
            text: "answer".to_string(),
        },
    );
    let mut extended = record_for(&item);
    extended["payload"]["unexpected_member"] = Value::String("injected".to_string());

    assert_eq!(
        stored_payload_fingerprint(&record_for(&item)),
        intended(&item)
    );
    assert_ne!(stored_payload_fingerprint(&extended), intended(&item));
    // The typed decoder still accepts the record, which is why the fingerprint has to catch it.
    let decoded = decode_rollout_line(extended).expect("record still decodes");
    assert_eq!(json(&decoded.item), json(&item));
}

/// Host-stamped tool results go through the same recording API the runtime uses.
#[test]
fn host_stamped_tool_results_reach_the_record_and_stay_comparable() {
    let stamped = host_stamped_function_call_output();
    let plain = function_call_output();
    let record = record_for(&stamped);

    let metadata = payload_metadata(&json(&stamped));
    for field in ["cell_id", "executed_tool_calls", "tool_calls_complete"] {
        assert!(metadata.get(field).is_some(), "{field} must be stamped");
    }
    // Same intended durable data still matches its record ...
    assert_eq!(stored_payload_fingerprint(&record), intended(&stamped));
    // ... while the stamping stays distinguishable to the fingerprint.
    assert_ne!(intended(&stamped), intended(&plain));
    // Decoding keeps the evidence out: every stamped field is gone, leaving only an empty
    // metadata envelope, so the decoded item can no longer tell the two writes apart.
    let decoded = decode_rollout_line(record).expect("decode stamped");
    assert_eq!(
        payload_metadata(&json(&decoded.item)),
        serde_json::json!({})
    );
}

/// The envelope is the writer's, not the item's, so it must not enter the fingerprint.
#[test]
fn writer_assigned_envelope_members_are_outside_the_fingerprint() {
    let item = message(
        "user",
        ContentItem::InputText {
            text: "hello".to_string(),
        },
    );
    let mut later = record_for(&item);
    later["timestamp"] = Value::String("2027-01-01T00:00:00.000Z".to_string());
    later["ordinal"] = Value::Number(9_999.into());

    assert_eq!(
        stored_payload_fingerprint(&later),
        stored_payload_fingerprint(&record_for(&item))
    );
    assert_eq!(stored_payload_fingerprint(&later), intended(&item));
}

/// The fingerprint must accept the same intended durable data and reject different durable data.
///
/// The second half is the property a decode-normalized fingerprint cannot hold: `cell_id`,
/// `executed_tool_calls` and `tool_calls_complete` are written to the record but never read back,
/// so normalizing by decoding would make these two payloads indistinguishable even though the
/// writer put different bytes on disk.
#[test]
fn differing_durable_payloads_do_not_collide_through_fields_the_decoder_drops() {
    let mut other = lossy_metadata();
    other.cell_id = Some("a-different-cell".to_string());
    let changed = response_item(ResponseItem::Message {
        id: None,
        role: "assistant".to_string(),
        content: vec![ContentItem::OutputText {
            text: "carrying host metadata".to_string(),
        }],
        phase: None,
        internal_chat_message_metadata_passthrough: Some(other),
    });
    let original = message_with_lossy_metadata();

    // Same intended durable data is accepted.
    assert_eq!(
        stored_payload_fingerprint(&record_for(&original)),
        intended(&original)
    );
    // Different durable data stays different, even though both decode identically.
    assert_ne!(intended(&changed), intended(&original));
    assert_ne!(
        stored_payload_fingerprint(&record_for(&changed)),
        stored_payload_fingerprint(&record_for(&original))
    );
    let decoded_original = decode_rollout_line(record_for(&original)).expect("decode original");
    let decoded_changed = decode_rollout_line(record_for(&changed)).expect("decode changed");
    assert_eq!(json(&decoded_original.item), json(&decoded_changed.item));
}
