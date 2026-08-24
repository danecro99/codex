use super::*;
use crate::config::ConfigBuilder;
use codex_config::ConfigLayerStack;
use codex_config::ConfigRequirements;
use codex_config::ConfigRequirementsToml;
use codex_config::FeatureRequirementsToml;
use codex_config::RequirementSource;
use codex_config::Sourced;
use codex_config::config_toml::ConfigToml;
use codex_config::config_toml::ForcedChatgptWorkspaceIds;
use codex_config::types::AuthCredentialsStoreMode;
use codex_features::FeaturesToml;
use codex_login::AuthDotJson;
use codex_login::load_auth_dot_json;
use codex_login::save_auth;
use codex_protocol::auth::AuthMode;
use codex_protocol::config_types::ForcedLoginMethod;
use codex_utils_absolute_path::AbsolutePathBuf;
use codex_utils_home_dir::resolve_codex_auth_home;
use pretty_assertions::assert_eq;
use std::collections::BTreeMap;
use tempfile::TempDir;

#[test]
fn resolve_bootstrap_auth_keyring_backend_kind_uses_secret_auth_storage_feature()
-> std::io::Result<()> {
    let config_toml = ConfigToml {
        features: Some(FeaturesToml::from(BTreeMap::from([(
            "secret_auth_storage".to_string(),
            true,
        )]))),
        ..Default::default()
    };
    assert_eq!(
        resolve_bootstrap_auth_keyring_backend_kind(&config_toml_load_result(
            config_toml,
            /*feature_requirements*/ None,
        )?)?,
        AuthKeyringBackendKind::Secrets
    );

    let config_toml = ConfigToml {
        features: Some(FeaturesToml::from(BTreeMap::from([(
            "secret_auth_storage".to_string(),
            false,
        )]))),
        ..Default::default()
    };
    assert_eq!(
        resolve_bootstrap_auth_keyring_backend_kind(&config_toml_load_result(
            config_toml.clone(),
            /*feature_requirements*/ None,
        )?)?,
        AuthKeyringBackendKind::Direct
    );

    let requirements = Sourced::new(
        FeatureRequirementsToml {
            entries: BTreeMap::from([("secret_auth_storage".to_string(), true)]),
        },
        RequirementSource::Unknown,
    );
    assert_eq!(
        resolve_bootstrap_auth_keyring_backend_kind(&config_toml_load_result(
            config_toml,
            Some(requirements),
        )?)?,
        AuthKeyringBackendKind::Secrets
    );

    Ok(())
}

#[test]
fn managed_auth_restrictions_intersect_workspaces_and_fail_closed() {
    let config = ConfigToml {
        forced_login_method: None,
        forced_chatgpt_workspace_id: Some(ForcedChatgptWorkspaceIds::Multiple(vec![
            " denied ".to_string(),
            " allowed ".to_string(),
        ])),
        ..Default::default()
    };
    let mut requirements = ConfigRequirements {
        allowed_login_methods: Some(Sourced::new(
            vec![ForcedLoginMethod::Chatgpt],
            RequirementSource::Unknown,
        )),
        allowed_chatgpt_workspaces: Some(Sourced::new(
            vec!["allowed".to_string()],
            RequirementSource::Unknown,
        )),
        ..Default::default()
    };

    let bootstrap_config = ConfigTomlLoadResult {
        config_toml: config.clone(),
        config_layer_stack: ConfigLayerStack::new(
            Vec::new(),
            requirements.clone(),
            ConfigRequirementsToml::default(),
        )
        .expect("requirements should stack"),
    };
    let (_codex_home_dir, codex_home) = absolute_temp_home();
    let (_auth_home_dir, auth_home) = absolute_temp_home();
    let auth_config = bootstrap_auth_config(&codex_home, &auth_home, &bootstrap_config)
        .expect("policy should resolve");
    assert_eq!(auth_config.codex_home, codex_home.to_path_buf());
    assert_eq!(auth_config.auth_home, auth_home.to_path_buf());
    assert_eq!(auth_config.forced_login_method, None);
    assert!(auth_config.is_login_method_allowed(ForcedLoginMethod::Chatgpt));
    assert!(!auth_config.is_login_method_allowed(ForcedLoginMethod::Api));
    assert_eq!(
        auth_config.forced_chatgpt_workspace_id,
        Some(vec!["denied".to_string(), "allowed".to_string()])
    );
    assert_eq!(
        auth_config.effective_chatgpt_workspaces(),
        Some(vec!["allowed".to_string()])
    );

    requirements.allowed_chatgpt_workspaces =
        Some(Sourced::new(Vec::new(), RequirementSource::Unknown));
    let bootstrap_config = ConfigTomlLoadResult {
        config_toml: config,
        config_layer_stack: ConfigLayerStack::new(
            Vec::new(),
            requirements,
            ConfigRequirementsToml::default(),
        )
        .expect("requirements should stack"),
    };
    assert_eq!(
        bootstrap_auth_config(&codex_home, &auth_home, &bootstrap_config)
            .expect_err("ChatGPT-only policy without an allowed workspace must fail")
            .kind(),
        std::io::ErrorKind::PermissionDenied
    );
}

#[test]
fn bootstrap_auth_config_applies_managed_store_and_chatgpt_base_url() {
    let configured_store = AuthCredentialsStoreMode::File;
    let configured_url = "https://user.example/backend-api/";
    let managed_store = AuthCredentialsStoreMode::Keyring;
    let managed_url = "https://managed.example/backend-api/";
    let config_toml = ConfigToml {
        cli_auth_credentials_store: Some(configured_store),
        chatgpt_base_url: Some(configured_url.to_string()),
        ..Default::default()
    };
    let requirements = ConfigRequirements {
        cli_auth_credentials_store: Some(Sourced::new(managed_store, RequirementSource::Unknown)),
        chatgpt_base_url: Some(Sourced::new(
            managed_url.to_string(),
            RequirementSource::Unknown,
        )),
        ..Default::default()
    };
    let bootstrap_config = ConfigTomlLoadResult {
        config_toml,
        config_layer_stack: ConfigLayerStack::new(
            Vec::new(),
            requirements,
            ConfigRequirementsToml::default(),
        )
        .expect("requirements should stack"),
    };

    let (_codex_home_dir, codex_home) = absolute_temp_home();
    let (_auth_home_dir, auth_home) = absolute_temp_home();
    let auth_config = bootstrap_auth_config(&codex_home, &auth_home, &bootstrap_config)
        .expect("managed authentication settings should resolve");

    assert_eq!(auth_config.auth_credentials_store_mode, managed_store);
    assert_eq!(auth_config.chatgpt_base_url.as_deref(), Some(managed_url));
}

#[test]
fn resolved_shared_auth_home_is_used_by_parallel_file_auth_storage() -> anyhow::Result<()> {
    let (_first_codex_home_dir, first_codex_home) = absolute_temp_home();
    let (_second_codex_home_dir, second_codex_home) = absolute_temp_home();
    let (_auth_home_dir, auth_home) = absolute_temp_home();
    let auth_home_override = auth_home.join(".");
    let first_auth_home = resolve_codex_auth_home(&first_codex_home, Some(&auth_home_override))?;
    let second_auth_home = resolve_codex_auth_home(&second_codex_home, Some(&auth_home_override))?;
    assert_eq!(first_auth_home, auth_home);
    assert_eq!(second_auth_home, auth_home);

    let expected = AuthDotJson {
        auth_mode: Some(AuthMode::ApiKey),
        openai_api_key: Some("synthetic-test-key".to_string()),
        tokens: None,
        last_refresh: None,
        agent_identity: None,
        personal_access_token: None,
        bedrock_api_key: None,
    };
    save_auth(
        first_auth_home.as_path(),
        &expected,
        AuthCredentialsStoreMode::File,
        AuthKeyringBackendKind::default(),
    )?;

    std::thread::scope(|scope| {
        let first_read = scope.spawn(|| {
            load_auth_dot_json(
                first_auth_home.as_path(),
                AuthCredentialsStoreMode::File,
                AuthKeyringBackendKind::default(),
            )
        });
        let second_read = scope.spawn(|| {
            load_auth_dot_json(
                second_auth_home.as_path(),
                AuthCredentialsStoreMode::File,
                AuthKeyringBackendKind::default(),
            )
        });

        assert_eq!(
            first_read.join().expect("first storage thread")?,
            Some(expected.clone())
        );
        assert_eq!(
            second_read.join().expect("second storage thread")?,
            Some(expected)
        );
        Ok::<(), anyhow::Error>(())
    })?;

    assert!(auth_home.join("auth.json").exists());
    assert!(!first_codex_home.join("auth.json").exists());
    assert!(!second_codex_home.join("auth.json").exists());
    Ok(())
}

#[tokio::test]
async fn config_builder_preserves_pre_resolved_auth_home() -> std::io::Result<()> {
    let (_codex_home_dir, codex_home) = absolute_temp_home();
    let (_auth_home_dir, auth_home) = absolute_temp_home();

    let config = ConfigBuilder::default()
        .codex_home(codex_home.to_path_buf())
        .auth_home(auth_home.clone())
        .build()
        .await?;

    assert_eq!(config.codex_home, codex_home);
    assert_eq!(config.auth_home, auth_home);
    Ok(())
}

fn absolute_temp_home() -> (TempDir, AbsolutePathBuf) {
    let home = tempfile::tempdir().expect("temp home");
    let absolute_home = AbsolutePathBuf::from_absolute_path(
        home.path().canonicalize().expect("canonicalize temp home"),
    )
    .expect("absolute temp home");
    (home, absolute_home)
}

fn config_toml_load_result(
    config_toml: ConfigToml,
    feature_requirements: Option<Sourced<FeatureRequirementsToml>>,
) -> std::io::Result<ConfigTomlLoadResult> {
    let requirements = ConfigRequirements {
        feature_requirements,
        ..Default::default()
    };
    Ok(ConfigTomlLoadResult {
        config_toml,
        config_layer_stack: ConfigLayerStack::new(
            Vec::new(),
            requirements,
            ConfigRequirementsToml::default(),
        )?,
    })
}
