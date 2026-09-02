use codex_utils_absolute_path::AbsolutePathBuf;
use dirs::home_dir;
use std::path::Path;
use std::path::PathBuf;

/// Returns the path to the Codex configuration directory, which can be
/// specified by the `CODEX_HOME` environment variable. If not set, defaults to
/// `~/.codex`.
///
/// - If `CODEX_HOME` is set, the value must exist and be a directory. The
///   value will be canonicalized and this function will Err otherwise.
/// - If `CODEX_HOME` is not set, this function does not verify that the
///   directory exists.
pub fn find_codex_home() -> std::io::Result<AbsolutePathBuf> {
    let codex_home_env = std::env::var("CODEX_HOME")
        .ok()
        .filter(|val| !val.is_empty());
    find_codex_home_from_env(codex_home_env.as_deref())
}

/// Returns the directory used exclusively for Codex authentication state.
///
/// When `CODEX_AUTH_HOME` is unset, authentication state stays in `codex_home`.
/// When set, the directory must already exist and is canonicalized before use.
pub fn find_codex_auth_home(codex_home: &AbsolutePathBuf) -> std::io::Result<AbsolutePathBuf> {
    let codex_auth_home_env = match std::env::var("CODEX_AUTH_HOME") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "CODEX_AUTH_HOME must be valid Unicode",
            ));
        }
    };
    resolve_codex_auth_home(codex_home, codex_auth_home_env.as_deref().map(Path::new))
}

/// Resolves an optional authentication-home override for a Codex state root.
///
/// A supplied override must exist as a directory and is canonicalized before
/// it is returned. `None` preserves the supplied Codex state root exactly.
pub fn resolve_codex_auth_home(
    codex_home: &AbsolutePathBuf,
    codex_auth_home: Option<&Path>,
) -> std::io::Result<AbsolutePathBuf> {
    match codex_auth_home {
        Some(path) => {
            if path.as_os_str().is_empty() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "CODEX_AUTH_HOME must name an existing directory",
                ));
            }
            let metadata = std::fs::metadata(path).map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!(
                        "CODEX_AUTH_HOME points to {:?}, but that path does not exist",
                        path.display()
                    ),
                ),
                _ => std::io::Error::new(
                    err.kind(),
                    format!("failed to read CODEX_AUTH_HOME {:?}: {err}", path.display()),
                ),
            })?;

            if !metadata.is_dir() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "CODEX_AUTH_HOME points to {:?}, but that path is not a directory",
                        path.display()
                    ),
                ));
            }

            let canonical = path.canonicalize().map_err(|err| {
                std::io::Error::new(
                    err.kind(),
                    format!(
                        "failed to canonicalize CODEX_AUTH_HOME {:?}: {err}",
                        path.display()
                    ),
                )
            })?;
            AbsolutePathBuf::from_absolute_path(canonical)
        }
        None => Ok(codex_home.clone()),
    }
}

fn find_codex_home_from_env(codex_home_env: Option<&str>) -> std::io::Result<AbsolutePathBuf> {
    // Honor the `CODEX_HOME` environment variable when it is set to allow users
    // (and tests) to override the default location.
    match codex_home_env {
        Some(val) => {
            let path = PathBuf::from(val);
            let metadata = std::fs::metadata(&path).map_err(|err| match err.kind() {
                std::io::ErrorKind::NotFound => std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("CODEX_HOME points to {val:?}, but that path does not exist"),
                ),
                _ => std::io::Error::new(
                    err.kind(),
                    format!("failed to read CODEX_HOME {val:?}: {err}"),
                ),
            })?;

            if !metadata.is_dir() {
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("CODEX_HOME points to {val:?}, but that path is not a directory"),
                ))
            } else {
                let canonical = path.canonicalize().map_err(|err| {
                    std::io::Error::new(
                        err.kind(),
                        format!("failed to canonicalize CODEX_HOME {val:?}: {err}"),
                    )
                })?;
                AbsolutePathBuf::from_absolute_path(canonical)
            }
        }
        None => {
            let mut p = home_dir().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "Could not find home directory",
                )
            })?;
            p.push(".codex");
            AbsolutePathBuf::from_absolute_path(p)
        }
    }
}

#[cfg(test)]
fn find_codex_auth_home_from_env(
    codex_home: &AbsolutePathBuf,
    codex_auth_home_env: Option<&str>,
) -> std::io::Result<AbsolutePathBuf> {
    resolve_codex_auth_home(codex_home, codex_auth_home_env.map(Path::new))
}

#[cfg(test)]
mod tests {
    use super::find_codex_auth_home_from_env;
    use super::find_codex_home_from_env;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use dirs::home_dir;
    use pretty_assertions::assert_eq;
    use std::fs;
    use std::io::ErrorKind;
    use tempfile::TempDir;

    #[test]
    fn find_codex_home_env_missing_path_is_fatal() {
        let temp_home = TempDir::new().expect("temp home");
        let missing = temp_home.path().join("missing-codex-home");
        let missing_str = missing
            .to_str()
            .expect("missing codex home path should be valid utf-8");

        let err = find_codex_home_from_env(Some(missing_str)).expect_err("missing CODEX_HOME");
        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(
            err.to_string().contains("CODEX_HOME"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn find_codex_home_env_file_path_is_fatal() {
        let temp_home = TempDir::new().expect("temp home");
        let file_path = temp_home.path().join("codex-home.txt");
        fs::write(&file_path, "not a directory").expect("write temp file");
        let file_str = file_path
            .to_str()
            .expect("file codex home path should be valid utf-8");

        let err = find_codex_home_from_env(Some(file_str)).expect_err("file CODEX_HOME");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("not a directory"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn find_codex_home_env_valid_directory_canonicalizes() {
        let temp_home = TempDir::new().expect("temp home");
        let temp_str = temp_home
            .path()
            .to_str()
            .expect("temp codex home path should be valid utf-8");

        let resolved = find_codex_home_from_env(Some(temp_str)).expect("valid CODEX_HOME");
        let expected = temp_home
            .path()
            .canonicalize()
            .expect("canonicalize temp home");
        let expected = AbsolutePathBuf::from_absolute_path(expected).expect("absolute home");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn find_codex_home_without_env_uses_default_home_dir() {
        let resolved =
            find_codex_home_from_env(/*codex_home_env*/ None).expect("default CODEX_HOME");
        let mut expected = home_dir().expect("home dir");
        expected.push(".codex");
        let expected = AbsolutePathBuf::from_absolute_path(expected).expect("absolute home");
        assert_eq!(resolved, expected);
    }

    #[test]
    fn find_codex_auth_home_without_env_uses_codex_home() {
        let codex_home = TempDir::new().expect("temp Codex home");
        let codex_home = AbsolutePathBuf::from_absolute_path(
            codex_home
                .path()
                .canonicalize()
                .expect("canonicalize Codex home"),
        )
        .expect("absolute Codex home");

        let resolved =
            find_codex_auth_home_from_env(&codex_home, /*codex_auth_home_env*/ None)
                .expect("default auth home");

        assert_eq!(resolved, codex_home);
    }

    #[test]
    fn find_codex_auth_home_env_canonicalizes_shared_home_for_distinct_codex_homes() {
        let first_codex_home = TempDir::new().expect("first temp Codex home");
        let second_codex_home = TempDir::new().expect("second temp Codex home");
        let auth_home = TempDir::new().expect("temp auth home");
        let first_codex_home = AbsolutePathBuf::from_absolute_path(
            first_codex_home
                .path()
                .canonicalize()
                .expect("canonicalize first Codex home"),
        )
        .expect("absolute first Codex home");
        let second_codex_home = AbsolutePathBuf::from_absolute_path(
            second_codex_home
                .path()
                .canonicalize()
                .expect("canonicalize second Codex home"),
        )
        .expect("absolute second Codex home");
        let auth_home = auth_home.path().join(".");
        let auth_home_str = auth_home
            .to_str()
            .expect("auth home path should be valid utf-8");

        let first_resolved = find_codex_auth_home_from_env(&first_codex_home, Some(auth_home_str))
            .expect("valid CODEX_AUTH_HOME");
        let second_resolved =
            find_codex_auth_home_from_env(&second_codex_home, Some(auth_home_str))
                .expect("valid CODEX_AUTH_HOME");
        let expected = AbsolutePathBuf::from_absolute_path(
            auth_home.canonicalize().expect("canonicalize auth home"),
        )
        .expect("absolute auth home");

        assert_eq!(first_resolved, expected);
        assert_eq!(second_resolved, expected);
    }

    #[test]
    fn find_codex_auth_home_env_missing_path_is_fatal() {
        let codex_home = TempDir::new().expect("temp Codex home");
        let codex_home = AbsolutePathBuf::from_absolute_path(
            codex_home
                .path()
                .canonicalize()
                .expect("canonicalize Codex home"),
        )
        .expect("absolute Codex home");
        let missing = codex_home.join("missing-auth-home");
        let missing_str = missing
            .to_str()
            .expect("missing auth home path should be valid utf-8");

        let err = find_codex_auth_home_from_env(&codex_home, Some(missing_str))
            .expect_err("missing CODEX_AUTH_HOME");

        assert_eq!(err.kind(), ErrorKind::NotFound);
        assert!(
            err.to_string().contains("CODEX_AUTH_HOME"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn find_codex_auth_home_env_empty_value_is_fatal() {
        let codex_home = TempDir::new().expect("temp Codex home");
        let codex_home = AbsolutePathBuf::from_absolute_path(
            codex_home
                .path()
                .canonicalize()
                .expect("canonicalize Codex home"),
        )
        .expect("absolute Codex home");

        let err = find_codex_auth_home_from_env(&codex_home, Some(""))
            .expect_err("empty CODEX_AUTH_HOME");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("CODEX_AUTH_HOME"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn find_codex_auth_home_env_file_path_is_fatal() {
        let codex_home = TempDir::new().expect("temp Codex home");
        let codex_home = AbsolutePathBuf::from_absolute_path(
            codex_home
                .path()
                .canonicalize()
                .expect("canonicalize Codex home"),
        )
        .expect("absolute Codex home");
        let auth_file = codex_home.join("auth-home-file");
        fs::write(&auth_file, "not a directory").expect("write auth home file");
        let auth_file_str = auth_file
            .to_str()
            .expect("auth home file path should be valid utf-8");

        let err = find_codex_auth_home_from_env(&codex_home, Some(auth_file_str))
            .expect_err("file CODEX_AUTH_HOME");

        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(
            err.to_string().contains("not a directory"),
            "unexpected error: {err}"
        );
    }
}
