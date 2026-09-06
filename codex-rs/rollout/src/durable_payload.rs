//! Canonical fingerprint of the payload a rollout record durably carries.
//!
//! A producer that records a write intent before handing items to the writer, and a verifier that
//! re-checks that intent against the records on disk, must describe the same thing. Fingerprinting
//! the in-memory item on one side and a decoded item on the other cannot work: the persisted
//! encoding is deliberately not a serialization fixed point. `ResponseItem::Reasoning::content` is
//! dropped whenever it holds no `reasoning_text` and decodes back as `None`, which serializes again
//! as an explicit `null`.
//!
//! Normalizing the intent by decoding it would hide a second, worse problem. Fields that are
//! `skip_deserializing` on purpose, such as the host-owned tool-call evidence on
//! `InternalChatMessageMetadataPassthrough`, never survive a decode, so two different durable
//! payloads that differ only in those fields would fingerprint the same. The intent check would
//! then accept a record the writer never intended.
//!
//! Both sides therefore fingerprint the durable payload directly, through [`RolloutItem`]'s own
//! `Serialize` impl - the same one the canonical writer flattens into each record - and never
//! deserialize. Anti-forgery fields stay closed for reading while still being compared.

use serde_json::Map;
use serde_json::Value;

use crate::RolloutItem;

/// Record members owned by the writer rather than by the item's durable payload.
const ENVELOPE_FIELDS: [&str; 2] = ["timestamp", "ordinal"];

/// Canonical bytes for the payload `item` contributes to the record the writer will produce.
pub fn intended_payload_fingerprint(item: &RolloutItem) -> serde_json::Result<Vec<u8>> {
    Ok(canonical_bytes(&serde_json::to_value(item)?))
}

/// Canonical bytes for the payload a persisted record already carries.
///
/// `record` is one decoded JSONL line, still holding the writer-assigned envelope members that the
/// intent cannot know.
pub fn stored_payload_fingerprint(record: &Value) -> Vec<u8> {
    let Value::Object(fields) = record else {
        return canonical_bytes(record);
    };
    let payload = fields
        .iter()
        .filter(|(key, _)| !ENVELOPE_FIELDS.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<String, Value>>();
    canonical_bytes(&Value::Object(payload))
}

/// Encodes a JSON value so equal values always produce equal bytes.
///
/// Every node is type-tagged and every string and collection is length-prefixed, so no value can
/// be confused with a differently shaped one. Object members are emitted in sorted key order
/// because `serde_json`'s map ordering depends on the `preserve_order` feature, which feature
/// unification turns on or off depending on which crates share a build.
fn canonical_bytes(value: &Value) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_canonical(value, &mut bytes);
    bytes
}

fn write_canonical(value: &Value, bytes: &mut Vec<u8>) {
    match value {
        Value::Null => bytes.push(b'n'),
        Value::Bool(value) => {
            bytes.push(b'b');
            bytes.push(u8::from(*value));
        }
        Value::Number(number) => {
            // `to_string` keeps the record's exact digits under `serde_json/arbitrary_precision`
            // and is the shortest round-trip form without it.
            bytes.push(b'#');
            write_bytes(number.to_string().as_bytes(), bytes);
        }
        Value::String(value) => {
            bytes.push(b's');
            write_bytes(value.as_bytes(), bytes);
        }
        Value::Array(items) => {
            bytes.push(b'[');
            write_len(items.len(), bytes);
            for item in items {
                write_canonical(item, bytes);
            }
        }
        Value::Object(fields) => {
            bytes.push(b'{');
            write_len(fields.len(), bytes);
            let mut keys = fields.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for key in keys {
                write_bytes(key.as_bytes(), bytes);
                match fields.get(key) {
                    Some(value) => write_canonical(value, bytes),
                    None => bytes.push(b'n'),
                }
            }
        }
    }
}

fn write_bytes(value: &[u8], bytes: &mut Vec<u8>) {
    write_len(value.len(), bytes);
    bytes.extend_from_slice(value);
}

fn write_len(len: usize, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(&u64::try_from(len).unwrap_or(u64::MAX).to_le_bytes());
}
