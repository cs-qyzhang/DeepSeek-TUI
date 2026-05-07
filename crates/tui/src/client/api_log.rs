//! API call YAML logger — records every LLM HTTP request and its response
//! to `~/.deepseek/api-logs/` for debugging and audit purposes.
//!
//! **Opt-in by default.** Set `enable_api_log = true` in `config.toml` or
//! `DEEPSEEK_API_LOG_ENABLE=1` to activate.  When enabled, logs are written
//! synchronously via `std::fs::write` on a best-effort basis — a failed
//! write is silently ignored so a full disk or permission error never blocks
//! the turn.

use chrono::Utc;
use serde::Serialize;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::logging;

/// Environment variable to override the default log directory.
const LOG_DIR_ENV: &str = "DEEPSEEK_API_LOG_DIR";

/// Environment variable to enable/disable API logging (`1`/`true` = on,
/// `0`/`false` = off).  Overrides the `enable_api_log` config.toml setting.
/// When unset, the config value is used.
const LOG_ENABLE_ENV: &str = "DEEPSEEK_API_LOG_ENABLE";

/// Default max number of log files to keep. Oldest files are pruned when
/// the directory exceeds this count.
const MAX_LOG_FILES: usize = 200;

/// Runtime flag set from config at startup. Off by default.
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Called once at startup from `App::new()`. When the config file sets
/// `enable_api_log = true`, logging is activated for this session.
pub fn set_enabled(yes: bool) {
    ENABLED.store(yes, Ordering::Relaxed);
}

/// Structured log entry for one API call.
#[derive(Debug, Serialize)]
struct ApiCallLog<'a> {
    timestamp: String,
    model: &'a str,
    endpoint: &'a str,
    duration_ms: u64,
    request: RequestLog<'a>,
    response: ResponseLog<'a>,
}

#[derive(Debug, Serialize)]
struct RequestLog<'a> {
    body: &'a Value,
}

#[derive(Debug, Serialize)]
struct ResponseLog<'a> {
    status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    body: Option<&'a Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_note: Option<&'a str>,
}

/// Log an API call to disk as YAML.
///
/// For non-streaming calls, pass the parsed response JSON as `response_body`.
/// For streaming calls, pass `None` for `response_body` and optionally
/// set `stream_note` to a short description (e.g. "streaming SSE").
pub fn log_api_call(
    model: &str,
    endpoint: &str,
    request_body: &Value,
    response_status: u16,
    response_body: Option<&Value>,
    duration: Duration,
) {
    if !should_log() {
        return;
    }

    let log = ApiCallLog {
        timestamp: Utc::now().to_rfc3339(),
        model,
        endpoint,
        duration_ms: duration.as_millis() as u64,
        request: RequestLog {
            body: request_body,
        },
        response: ResponseLog {
            status: response_status,
            body: response_body,
            stream_note: if response_body.is_none() {
                Some("streaming — response body consumed via SSE")
            } else {
                None
            },
        },
    };

    let yaml = match serde_saphyr::to_string(&log) {
        Ok(y) => y,
        Err(e) => {
            logging::warn(format!("api_log: failed to serialize YAML: {e}"));
            return;
        }
    };

    let dir = api_log_dir();
    if let Err(e) = std::fs::create_dir_all(&dir) {
        logging::warn(format!("api_log: failed to create dir {}: {e}", dir.display()));
        return;
    }

    let ts = Utc::now().format("%Y%m%dT%H%M%S");
    let model_slug = model.replace(['/', '\\', ':', ' '], "_");
    let filename = format!("{ts}_{model_slug}.yaml");
    let path = dir.join(&filename);

    if let Err(e) = std::fs::write(&path, yaml.as_bytes()) {
        logging::warn(format!(
            "api_log: failed to write {}: {e}",
            path.display()
        ));
        return;
    }

    logging::info(format!("api_log: wrote {}", path.display()));

    // Best-effort prune.
    let _ = prune_old_logs(&dir);
}

fn api_log_dir() -> PathBuf {
    if let Ok(dir) = std::env::var(LOG_DIR_ENV)
        && !dir.is_empty()
    {
        return PathBuf::from(dir);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".deepseek")
        .join("api-logs")
}

/// Check whether API logging is active: env var wins over config.
///
/// - `DEEPSEEK_API_LOG_ENABLE=1` → on (overrides config)
/// - `DEEPSEEK_API_LOG_ENABLE=0` → off (overrides config)
/// - unset → use the config value stored in `ENABLED`
fn should_log() -> bool {
    if let Ok(v) = std::env::var(LOG_ENABLE_ENV) {
        return matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        );
    }
    ENABLED.load(Ordering::Relaxed)
}

/// Delete oldest log files when the directory exceeds `MAX_LOG_FILES`.
fn prune_old_logs(dir: &std::path::Path) -> std::io::Result<()> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "yaml"))
        .collect();

    if entries.len() <= MAX_LOG_FILES {
        return Ok(());
    }

    // Sort by modified time, oldest first.
    entries.sort_by_key(|p| {
        std::fs::metadata(p)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
    });

    let to_remove = entries.len() - MAX_LOG_FILES;
    for path in entries.iter().take(to_remove) {
        let _ = std::fs::remove_file(path);
    }

    if to_remove > 0 {
        logging::info(format!("api_log: pruned {to_remove} old log file(s)"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Serialize env-var-dependent tests to avoid races from parallel runners.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn log_api_call_writes_yaml_file() {
        let _lock = ENV_LOCK.lock().unwrap();
        set_enabled(true);
        let tmp = TempDir::new().unwrap();
        unsafe { std::env::remove_var(LOG_ENABLE_ENV) };
        unsafe { std::env::set_var(LOG_DIR_ENV, tmp.path().as_os_str()) };

        let body = serde_json::json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        });

        log_api_call(
            "deepseek-v4-flash",
            "chat/completions",
            &body,
            200,
            Some(&serde_json::json!({"id": "chatcmpl-123", "choices": []})),
            Duration::from_millis(1234),
        );

        // Should have created exactly one .yaml file.
        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |e| e == "yaml"))
            .collect();
        assert_eq!(entries.len(), 1, "expected one log file");

        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(content.contains("model: deepseek-v4-flash"));
        assert!(content.contains("duration_ms: 1234"));
        assert!(content.contains("status: 200"));
        assert!(content.contains("chatcmpl-123"));
    }

    #[test]
    fn log_streaming_call_omits_response_body() {
        let _lock = ENV_LOCK.lock().unwrap();
        set_enabled(true);
        let tmp = TempDir::new().unwrap();
        unsafe { std::env::remove_var(LOG_ENABLE_ENV) };
        unsafe { std::env::set_var(LOG_DIR_ENV, tmp.path().as_os_str()) };

        let body = serde_json::json!({"model": "deepseek-v4-pro"});
        log_api_call(
            "deepseek-v4-pro",
            "chat/completions",
            &body,
            200,
            None, // streaming, no captured body
            Duration::from_millis(500),
        );

        let entries: Vec<_> = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |e| e == "yaml"))
            .collect();
        assert_eq!(entries.len(), 1);

        let content = std::fs::read_to_string(entries[0].path()).unwrap();
        assert!(content.contains("stream_note:"));
        assert!(content.contains("streaming"));
    }

    #[test]
    fn disabled_by_env_skips_write() {
        let _lock = ENV_LOCK.lock().unwrap();
        set_enabled(true); // config says yes, but env var overrides with 0
        let tmp = TempDir::new().unwrap();
        unsafe { std::env::remove_var(LOG_DIR_ENV) };
        unsafe { std::env::set_var(LOG_DIR_ENV, tmp.path().as_os_str()) };
        unsafe { std::env::set_var(LOG_ENABLE_ENV, "0") };

        let body = serde_json::json!({"model": "test"});
        log_api_call("test", "/", &body, 200, None, Duration::from_millis(1));

        // Should be empty.
        let count = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |e| e == "yaml"))
            .count();
        assert_eq!(count, 0);
    }

    #[test]
    fn prune_removes_oldest_files() {
        let tmp = TempDir::new().unwrap();
        // Create more than MAX_LOG_FILES dummy yaml files.
        for i in 0..(MAX_LOG_FILES + 5) {
            let path = tmp.path().join(format!("log_{i:05}.yaml"));
            std::fs::write(&path, b"dummy").unwrap();
            // Stagger mtimes so sorting is deterministic: oldest first.
            std::thread::sleep(std::time::Duration::from_millis(1));
        }

        let count_before = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |e| e == "yaml"))
            .count();
        assert_eq!(count_before, MAX_LOG_FILES + 5);

        prune_old_logs(tmp.path()).unwrap();

        let count_after = std::fs::read_dir(tmp.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |e| e == "yaml"))
            .count();
        assert_eq!(count_after, MAX_LOG_FILES);
    }
}