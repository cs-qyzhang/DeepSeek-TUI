//! Full-project code injection command: `/inject-full-codes`
//!
//! Walks the workspace directory using the `ignore` crate (respecting
//! `.gitignore`, `.ignore`, `.agentignore`, `.deepseekignore`) and collects all source
//! code and documentation files. Each file's full content is read and
//! formatted as a Markdown code fence, then injected into the prompt as
//! a user message. Designed for small-to-medium projects leveraging
//! DeepSeek's 1M-token context window.

use ignore::WalkBuilder;
use std::path::Path;

use crate::tui::app::{App, AppAction};

use super::CommandResult;

/// File extensions considered "project source or docs."
const PROJECT_EXTENSIONS: &[&str] = &[
    // Rust
    "rs", "toml",
    // Configuration
    "json", "yaml", "yml", "lock",
    // Documentation
    "md", "txt", "rst",
    // Scripts
    "sh", "bash", "zsh", "py", "rb", "pl",
    // Web
    "js", "jsx", "ts", "tsx", "html", "css", "scss", "sass", "less",
    // Systems
    "c", "cc", "cpp", "h", "hpp", "go", "java", "kt", "swift",
    // CI/Config
    "cfg", "ini", "env", "example",
    // Data / Query
    "sql", "graphql", "proto",
    // Misc text formats
    "svg", "xml",
];

/// Hard cap on the total bytes of file content collected.
/// ~800 KB ≈ 200K tokens at ~4 bytes/token, leaving ~800K tokens for
/// the system prompt, conversation history, and model response.
const MAX_TOTAL_BYTES: usize = 800_000;

/// Walk the workspace and inject every project file into the prompt.
pub fn inject_full_codes(app: &mut App) -> CommandResult {
    let workspace = app.workspace.clone();

    if !workspace.is_dir() {
        return CommandResult::error(format!(
            "Workspace is not a directory: {}",
            workspace.display()
        ));
    }

    let mut files: Vec<(String, String)> = Vec::new(); // (relative path, content)
    let mut total_bytes: usize = 0;
    let mut skipped_count: usize = 0;

    let mut builder = WalkBuilder::new(&workspace);
    builder
        .hidden(true) // visit hidden files (for .env.example etc) but not .git
        .follow_links(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true);
    // Also honor project-specific ignore files if present.
    let _ = builder.add_custom_ignore_filename(".agentignore");
    let _ = builder.add_custom_ignore_filename(".deepseekignore");

    for entry in builder.build().flatten() {
        // Only regular files, no symlinks or special files.
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();

        // Must have a recognized extension.
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext_lower = ext.to_ascii_lowercase();
        if !PROJECT_EXTENSIONS.contains(&ext_lower.as_str()) {
            continue;
        }

        // Read file contents. Skip binary / unreadable files.
        let Ok(content) = std::fs::read_to_string(path) else {
            continue;
        };
        // Skip empty files — no signal.
        if content.trim().is_empty() {
            continue;
        }

        // Check budget. If this file would push us over, skip it but keep
        // trying smaller files in case the walk yields them later.
        if total_bytes + content.len() > MAX_TOTAL_BYTES {
            skipped_count += 1;
            continue;
        }

        let rel = path.strip_prefix(&workspace).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");

        total_bytes += content.len();
        files.push((rel_str.to_string(), content));
    }

    if files.is_empty() {
        return CommandResult::message(
            "No project files found to inject. Check that the workspace contains \
             source or documentation files with recognized extensions.",
        );
    }

    // Sort by path for deterministic output (helps with prefix caching).
    files.sort_by(|a, b| a.0.cmp(&b.0));

    // Build the injection message.
    let mut msg = String::with_capacity(total_bytes + files.len() * 64);
    msg.push_str("## Full Project Code Injection\n\n");
    msg.push_str(&format!(
        "Workspace: {}\n",
        workspace.display()
    ));
    msg.push_str(&format!(
        "Files included: {} (~{} KB total)\n\n",
        files.len(),
        total_bytes / 1024
    ));
    msg.push_str("The following is the complete source code and documentation \
                   for this project. Each file is shown with its workspace-relative \
                   path and full contents.\n\n");
    msg.push_str("---\n\n");

    for (path, content) in &files {
        let lang = ext_to_language(path);
        msg.push_str(&format!("### `{path}`\n\n```{lang}\n{content}\n```\n\n"));
    }

    let skipped_note = if skipped_count > 0 {
        format!(
            " ({} file(s) skipped due to size budget)",
            skipped_count
        )
    } else {
        String::new()
    };

    msg.push_str("---\n");
    msg.push_str(&format!(
        "*Injected {} files, ~{} KB total.{}\n*",
        files.len(),
        total_bytes / 1024,
        skipped_note
    ));

    CommandResult::with_message_and_action(
        format!(
            "Injected {} files (~{} KB) into context{}",
            files.len(),
            total_bytes / 1024,
            skipped_note
        ),
        AppAction::SendMessage(msg),
    )
}

/// Map a file extension (or path) to a markdown code-fence language tag.
fn ext_to_language(path: &str) -> &'static str {
    let p = Path::new(path);
    let ext = p
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    match ext.to_ascii_lowercase().as_str() {
        "rs" => "rust",
        "toml" => "toml",
        "json" => "json",
        "yaml" | "yml" => "yaml",
        "md" => "markdown",
        "txt" | "rst" | "cfg" | "ini" | "env" | "example" | "lock" => "text",
        "sh" | "bash" | "zsh" => "bash",
        "py" => "python",
        "rb" => "ruby",
        "pl" => "perl",
        "js" => "javascript",
        "jsx" => "javascript",
        "ts" => "typescript",
        "tsx" => "typescript",
        "html" => "html",
        "css" => "css",
        "scss" | "sass" => "scss",
        "less" => "less",
        "c" => "c",
        "cc" | "cpp" => "cpp",
        "h" | "hpp" => "cpp",
        "go" => "go",
        "java" => "java",
        "kt" => "kotlin",
        "swift" => "swift",
        "sql" => "sql",
        "graphql" => "graphql",
        "proto" => "protobuf",
        "svg" => "xml",
        "xml" => "xml",
        _ => "", // plain text, no language tag
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::tui::app::{App, TuiOptions};
    use std::fs;
    use tempfile::TempDir;

    fn create_test_app_in(tmpdir: &TempDir) -> App {
        let options = TuiOptions {
            model: "deepseek-v4-pro".to_string(),
            workspace: tmpdir.path().to_path_buf(),
            config_path: None,
            config_profile: None,
            allow_shell: false,
            use_alt_screen: true,
            use_mouse_capture: false,
            use_bracketed_paste: true,
            max_subagents: 1,
            skills_dir: tmpdir.path().join("skills"),
            memory_path: tmpdir.path().join("memory.md"),
            notes_path: tmpdir.path().join("notes.txt"),
            mcp_config_path: tmpdir.path().join("mcp.json"),
            use_memory: false,
            start_in_agent_mode: false,
            skip_onboarding: true,
            yolo: false,
            resume_session_id: None,
            initial_input: None,
        };
        App::new(options, &Config::default())
    }

    #[test]
    fn inject_empty_workspace_returns_message() {
        let tmpdir = TempDir::new().unwrap();
        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);
        assert!(result.message.is_some());
        let msg = result.message.unwrap();
        assert!(
            msg.contains("No project files found"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn inject_collects_source_and_doc_files() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(tmpdir.path().join("README.md"), "# My Project").unwrap();
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        assert!(result.message.is_some());
        let status = result.message.unwrap();
        assert!(status.contains("Injected 3 files"), "got: {status}");

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(content.contains("### `main.rs`"));
                assert!(content.contains("fn main() {}"));
                assert!(content.contains("### `README.md`"));
                assert!(content.contains("# My Project"));
                assert!(content.contains("### `Cargo.toml`"));
                assert!(content.contains("[package]"));
                assert!(content.contains("```rust"));
                assert!(content.contains("```toml"));
                assert!(content.contains("```markdown"));
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_respects_agentignore() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join(".agentignore"), "excluded/\nsecret.py\n").unwrap();
        fs::create_dir(tmpdir.path().join("excluded")).unwrap();
        fs::write(tmpdir.path().join("excluded/hidden.rs"), "fn hidden() {}").unwrap();
        fs::write(tmpdir.path().join("secret.py"), "print('secret')").unwrap();
        fs::write(tmpdir.path().join("visible.py"), "print('visible')").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(
                    content.contains("visible.py"),
                    "visible file should be included; content: {content}"
                );
                assert!(
                    !content.contains("secret.py"),
                    ".agentignore-listed file should be excluded; content: {content}"
                );
                assert!(
                    !content.contains("hidden.rs"),
                    ".agentignore-listed directory should be excluded; content: {content}"
                );
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_respects_gitignore() {
        let tmpdir = TempDir::new().unwrap();
        // `.ignore` is honored by the ignore crate even outside a git repo.
        fs::write(tmpdir.path().join(".ignore"), "ignored/\n").unwrap();
        fs::create_dir(tmpdir.path().join("ignored")).unwrap();
        fs::write(tmpdir.path().join("ignored/secret.rs"), "fn secret() {}").unwrap();
        fs::write(tmpdir.path().join("visible.rs"), "fn visible() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(content.contains("visible.rs"), "content: {content}");
                assert!(
                    !content.contains("secret.rs"),
                    "ignored file leaked: {content}"
                );
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_skips_empty_files() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join("empty.rs"), "").unwrap();
        fs::write(tmpdir.path().join("real.rs"), "fn real() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(content.contains("real.rs"));
                assert!(!content.contains("empty.rs"));
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_skips_unsupported_extensions() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join("image.png"), "fake-png-data").unwrap();
        fs::write(tmpdir.path().join("lib.rs"), "pub fn x() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(content.contains("lib.rs"));
                assert!(!content.contains("image.png"));
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn ext_to_language_maps_correctly() {
        assert_eq!(ext_to_language("main.rs"), "rust");
        assert_eq!(ext_to_language("Cargo.toml"), "toml");
        assert_eq!(ext_to_language("README.md"), "markdown");
        assert_eq!(ext_to_language("script.py"), "python");
        assert_eq!(ext_to_language("app.js"), "javascript");
        assert_eq!(ext_to_language("types.ts"), "typescript");
        assert_eq!(ext_to_language("style.css"), "css");
        assert_eq!(ext_to_language("query.sql"), "sql");
        assert_eq!(ext_to_language("doc.txt"), "text");
        assert_eq!(ext_to_language("unknown.xyz"), "");
    }
}
