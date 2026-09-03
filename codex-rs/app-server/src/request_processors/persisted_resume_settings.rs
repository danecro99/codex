use codex_protocol::config_types::ApprovalsReviewer;
use codex_protocol::models::ActivePermissionProfile;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::TurnContextItem;
use codex_rollout::RolloutItem;

#[derive(Debug, PartialEq, Eq)]
pub(super) struct PersistedResumeSettings {
    pub(super) approval_policy: AskForApproval,
    pub(super) approvals_reviewer: Option<ApprovalsReviewer>,
    pub(super) active_permission_profile: Option<ActivePermissionProfile>,
}

pub(super) fn latest_persisted_resume_settings(
    history: &[RolloutItem],
) -> Option<PersistedResumeSettings> {
    scan_persisted_resume_settings(history.iter().collect::<Vec<_>>().as_slice())
}

/// Resolves persisted settings for a resume whose replay may be a checkpoint suffix.
///
/// A checkpoint hit returns only the appended suffix, so the reconstructed reference turn context
/// stored with the checkpoint stands in for the older records the suffix no longer carries. It is
/// older than every suffix record, so a newer suffix `TurnContext` still wins.
pub(super) fn latest_persisted_resume_settings_with_checkpoint(
    history: &[RolloutItem],
    checkpoint_context: Option<&TurnContextItem>,
) -> Option<PersistedResumeSettings> {
    let checkpoint_item = checkpoint_context.cloned().map(RolloutItem::TurnContext);
    let mut items = Vec::with_capacity(history.len().saturating_add(1));
    items.extend(checkpoint_item.as_ref());
    items.extend(history.iter());
    scan_persisted_resume_settings(items.as_slice())
}

fn scan_persisted_resume_settings(history: &[&RolloutItem]) -> Option<PersistedResumeSettings> {
    history
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, item)| match item {
            RolloutItem::TurnContext(turn_context) => Some(PersistedResumeSettings {
                approval_policy: turn_context.approval_policy,
                approvals_reviewer: turn_context.approvals_reviewer.or_else(|| {
                    history[..index].iter().rev().find_map(|item| match item {
                        RolloutItem::TurnContext(turn_context) => turn_context.approvals_reviewer,
                        RolloutItem::EventMsg(EventMsg::ThreadSettingsApplied(event)) => {
                            Some(event.thread_settings.approvals_reviewer)
                        }
                        _ => None,
                    })
                }),
                active_permission_profile: turn_context.active_permission_profile.clone(),
            }),
            RolloutItem::EventMsg(EventMsg::ThreadSettingsApplied(event)) => {
                Some(PersistedResumeSettings {
                    approval_policy: event.thread_settings.approval_policy,
                    approvals_reviewer: Some(event.thread_settings.approvals_reviewer),
                    active_permission_profile: event
                        .thread_settings
                        .active_permission_profile
                        .clone(),
                })
            }
            _ => None,
        })
}

#[cfg(test)]
#[path = "persisted_resume_settings_tests.rs"]
mod tests;
