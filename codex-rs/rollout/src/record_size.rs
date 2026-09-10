//! Validate complete canonical batches before publishing a write intent or queueing records.

use std::io;
use std::io::Write;

use crate::MAX_CANONICAL_ROLLOUT_RECORD_BYTES;
use crate::RolloutItem;
use crate::recorder::RolloutLineRef;

/// Reject oversized items before a batch can become partially durable.
///
/// Reserve the full writer envelope (millisecond UTC timestamp and the largest ordinal) before
/// the exact ordinal is assigned. This bounds storage bytes, not model tokens or the fingerprint's
/// binary encoding. Counting borrows items and does not allocate another record-sized buffer.
pub fn validate_canonical_rollout_items(items: &[RolloutItem]) -> io::Result<()> {
    for (index, item) in items.iter().enumerate() {
        let mut counter = RecordSize { bytes: 0 };
        let line = RolloutLineRef {
            timestamp: "9999-12-31T23:59:59.999Z".to_string(),
            ordinal: Some(u64::MAX),
            item,
        };
        serde_json::to_writer(&mut counter, &line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("canonical rollout batch item {index} failed size preflight: {err}"),
            )
        })?;
        counter.write_all(b"\n")?;
    }
    Ok(())
}

struct RecordSize {
    bytes: usize,
}

impl Write for RecordSize {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self.bytes.saturating_add(bytes.len());
        if next > MAX_CANONICAL_ROLLOUT_RECORD_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "canonical rollout record exceeds {MAX_CANONICAL_ROLLOUT_RECORD_BYTES} bytes including its writer envelope and newline (observed at least {next})"
                ),
            ));
        }
        self.bytes = next;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
