use super::*;
use crate::cache::CLOUD_CONFIG_BUNDLE_CACHE_FILENAME;
use codex_config::ManagedAuthPolicy;
use codex_config::types::AuthCredentialsStoreMode;
use codex_login::AuthKeyringBackendKind;
use pretty_assertions::assert_eq;
use tempfile::tempdir;

#[test]
fn auth_config_keeps_cloud_config_cache_under_codex_home() {
    let codex_home = tempdir().expect("temp Codex home");
    let auth_home = tempdir().expect("temp auth home");
    let auth_config = AuthConfig {
        codex_home: codex_home.path().to_path_buf(),
        auth_home: auth_home.path().to_path_buf(),
        auth_credentials_store_mode: AuthCredentialsStoreMode::Ephemeral,
        keyring_backend_kind: AuthKeyringBackendKind::default(),
        forced_login_method: None,
        chatgpt_base_url: None,
        forced_chatgpt_workspace_id: None,
        managed_auth_policy: ManagedAuthPolicy::default(),
        auth_route_config: codex_login::test_support::transport_default_auth_route_config(),
    };

    assert_eq!(
        auth_config
            .codex_home
            .join(CLOUD_CONFIG_BUNDLE_CACHE_FILENAME),
        codex_home.path().join(CLOUD_CONFIG_BUNDLE_CACHE_FILENAME)
    );
    assert_ne!(auth_config.codex_home, auth_config.auth_home);
}
