//! Full-project code injection command: `/inject-full-codes`
//!
//! Walks the workspace directory using the `ignore` crate (respecting
//! `.gitignore`, `.ignore`, `.deepseekignore`) and collects all source.
//! If a `.agentsee` file exists at the workspace root, it acts as an
//! include filter (gitignore syntax, but patterns specify what to include;
//! `!` patterns specify what to force-exclude).
//! code and documentation files. Each file's full content is read and
//! formatted as a Markdown code fence, then injected into the prompt as
//! a user message. Designed for small-to-medium projects leveraging
//! DeepSeek's 1M-token context window.

use glob::Pattern;
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

/// Parsed `.agentsee` include filter.
///
/// Gitignore syntax, but inverted: patterns specify what to **include**,
/// and lines starting with `!` specify what to **force-exclude** (taking
/// priority over include patterns).  Pattern order matters: files matching
/// earlier lines are given higher priority and will be included first
/// when the budget is tight.  When no `.agentsee` file exists, every file
/// passes with equal (lowest) priority.
struct Agentsee {
    includes: Vec<Pattern>,
    excludes: Vec<Pattern>,
}

impl Agentsee {
    /// Load `.agentsee` from the workspace root.  Returns `None` when the
    /// file is missing or empty — callers should include all files.
    fn load(workspace: &Path) -> Option<Self> {
        let path = workspace.join(".agentsee");
        let content = std::fs::read_to_string(&path).ok()?;
        if content.trim().is_empty() {
            return None;
        }

        let mut includes = Vec::new();
        let mut excludes = Vec::new();

        for raw in content.lines() {
            let line = raw.trim();
            // Skip blank lines and comments.
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (negated, pat_str) = if let Some(rest) = line.strip_prefix('!') {
                (true, rest.trim())
            } else {
                (false, line)
            };

            // Convert gitignore-style glob to standard glob syntax.
            let glob = gitignore_to_glob(pat_str);
            let Ok(pattern) = Pattern::new(&glob) else {
                continue;
            };

            if negated {
                excludes.push(pattern);
            } else {
                includes.push(pattern);
            }
        }

        Some(Self { includes, excludes })
    }

    /// Returns the priority index (0 = highest) for a workspace-relative
    /// path, or `None` if it should be excluded.  When `.agentsee` is
    /// absent, every file passes with max priority.
    fn priority(&self, rel: &str) -> Option<usize> {
        // Exclude patterns take priority.
        if self.excludes.iter().any(|p| p.matches(rel)) {
            return None;
        }
        // Return the index of the first matching include pattern.
        self.includes
            .iter()
            .position(|p| p.matches(rel))
    }
}

/// Convert a gitignore-style glob pattern to a standard glob pattern
/// compatible with the `glob` crate.
///
/// Transformations:
/// - Trailing `/` → `/**` (match directory and all descendants)
/// - Leading `/` is stripped (gitignore anchors; we always match relative
///   to the workspace root)
/// - `**` is already valid glob syntax
/// - `*` and `?` are already valid glob syntax
fn gitignore_to_glob(pat: &str) -> String {
    let pat = pat.trim();

    // Trailing `/` means "directory and everything under it".
    if let Some(prefix) = pat.strip_suffix('/') {
        return format!("{prefix}/**");
    }

    // Strip leading `/` — gitignore uses it to anchor to root, but our
    // relative paths are already anchored.
    let pat = pat.strip_prefix('/').unwrap_or(pat);

    // If the pattern doesn't start with `*` or `**`, prepend `**/` so it
    // matches at any depth (gitignore default behaviour for non-anchored
    // patterns).
    if !pat.starts_with('*') {
        format!("**/{pat}")
    } else {
        pat.to_string()
    }
}

/// Result of collecting project files for injection.
struct InjectPlan {
    /// The formatted injection message text.
    message: String,
    /// Number of files included.
    file_count: usize,
    /// Total bytes of file contents (not including markdown framing).
    total_bytes: usize,
    /// Number of files skipped due to budget.
    skipped_count: usize,
    /// Workspace-relative paths of included files (sorted).
    files: Vec<String>,
}

/// Walk the workspace and build the injection message text.
/// Returns `None` when no project files are found.
///
/// When a `.agentsee` file exists, files are collected in priority order:
/// files matching earlier patterns are read first.  When budget is tight,
/// later (less important) files are naturally skipped.
fn build_injection_message(workspace: &Path) -> Option<InjectPlan> {
    if !workspace.is_dir() {
        return None;
    }

    let agentsee = Agentsee::load(workspace);

    // Pass 1: collect candidate paths with their priority.
    // (priority, rel_path, abs_path)
    let mut candidates: Vec<(usize, String, std::path::PathBuf)> = Vec::new();

    let mut builder = WalkBuilder::new(workspace);
    builder
        .hidden(true)
        .follow_links(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(true);
    let _ = builder.add_custom_ignore_filename(".deepseekignore");

    for entry in builder.build().flatten() {
        if !entry.file_type().is_some_and(|ft| ft.is_file()) {
            continue;
        }
        let path = entry.path();

        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let ext_lower = ext.to_ascii_lowercase();
        if !PROJECT_EXTENSIONS.contains(&ext_lower.as_str()) {
            continue;
        }

        let rel = path.strip_prefix(workspace).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");

        // Determine priority: lower = more important.
        let priority = if let Some(ref see) = agentsee {
            // If the file matches no include pattern, skip it.
            // `None` means excluded (either by `!` or not matching any include).
            let Some(p) = see.priority(&rel_str) else {
                continue;
            };
            p
        } else {
            // No .agentsee → all files equal (lowest priority).
            usize::MAX
        };

        candidates.push((priority, rel_str.to_string(), path.to_path_buf()));
    }

    if candidates.is_empty() {
        return None;
    }

    // Sort by priority (lower first), then by path for deterministic output.
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Pass 2: read files in priority order, accumulating up to budget.
    let mut files: Vec<(String, String)> = Vec::new(); // (relative path, content)
    let mut total_bytes: usize = 0;
    let mut skipped_count: usize = 0;

    for (_priority, rel_str, abs_path) in &candidates {
        let Ok(content) = std::fs::read_to_string(abs_path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        if total_bytes + content.len() > MAX_TOTAL_BYTES {
            skipped_count += 1;
            continue;
        }

        total_bytes += content.len();
        files.push((rel_str.clone(), content));
    }

    let mut msg = String::with_capacity(total_bytes + files.len() * 64);
    msg.push_str("## Full Project Code Injection\n\n");
    msg.push_str(&format!("Workspace: {}\n", workspace.display()));
    msg.push_str(&format!(
        "Files included: {} (~{} KB total)\n\n",
        files.len(),
        total_bytes / 1024
    ));
    msg.push_str(
        "The following is the complete source code and documentation \
         for this project. Each file is shown with its workspace-relative \
         path and full contents.\n\n",
    );
    msg.push_str("---\n\n");

    for (path, content) in &files {
        let lang = ext_to_language(path);
        msg.push_str(&format!("### `{path}`\n\n```{lang}\n{content}\n```\n\n"));
    }

    let skipped_note = if skipped_count > 0 {
        format!(" ({} file(s) skipped due to size budget)", skipped_count)
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

    let file_paths: Vec<String> = files.iter().map(|(p, _)| p.clone()).collect();

    Some(InjectPlan {
        message: msg,
        file_count: files.len(),
        total_bytes,
        skipped_count,
        files: file_paths,
    })
}

/// Walk the workspace and inject every project file into the prompt.
pub fn inject_full_codes(app: &mut App) -> CommandResult {
    let Some(plan) = build_injection_message(&app.workspace) else {
        return CommandResult::message(
            "No project files found to inject. Check that the workspace contains \
             source or documentation files with recognized extensions.",
        );
    };

    let skipped_note = if plan.skipped_count > 0 {
        format!(
            " ({} file(s) skipped due to size budget)",
            plan.skipped_count
        )
    } else {
        String::new()
    };

    CommandResult::with_message_and_action(
        format!(
            "Injected {} files (~{} KB) into context{}",
            plan.file_count,
            plan.total_bytes / 1024,
            skipped_note
        ),
        AppAction::SendMessage(plan.message),
    )
}

/// Dry-run the injection and estimate how many tokens the full message
/// would consume. Does NOT send any message or modify the conversation.
pub fn full_codes_tokens(app: &App) -> CommandResult {
    let Some(plan) = build_injection_message(&app.workspace) else {
        return CommandResult::message(
            "No project files found for estimation. Check that the workspace \
             contains source or documentation files with recognized extensions.",
        );
    };

    // Conservative token estimate: ~4 chars per token for English / code.
    let char_count = plan.message.chars().count();
    let token_estimate = char_count.div_ceil(4);
    let kb = plan.total_bytes / 1024;

    let skipped_line = if plan.skipped_count > 0 {
        format!("\nFiles skipped (budget): {}", plan.skipped_count)
    } else {
        String::new()
    };

    let mut file_list = String::new();
    for f in &plan.files {
        file_list.push_str(&format!("  {f}\n"));
    }

    CommandResult::message(format!(
        "Full Codes Token Estimate\n\
         Workspace: {}\n\
         Files: {}\n\
         Content size: ~{} KB\n\
         Message chars: {}\n\
         Estimated tokens: ~{}  (~4 chars/token heuristic){}\n\
         \n\
         Files that would be injected:\n\
         {}",
        app.workspace.display(),
        plan.file_count,
        kb,
        char_count,
        token_estimate,
        skipped_line,
        file_list,
    ))
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
    fn inject_respects_agentsee() {
        let tmpdir = TempDir::new().unwrap();
        // *.rs first (higher priority), *.py second, !excluded/ excludes
        fs::write(
            tmpdir.path().join(".agentsee"),
            "*.rs\n*.py\n!excluded/\n",
        )
        .unwrap();
        fs::create_dir(tmpdir.path().join("excluded")).unwrap();
        fs::write(tmpdir.path().join("excluded/hidden.rs"), "fn hidden() {}").unwrap();
        fs::write(tmpdir.path().join("secret.py"), "print('secret')").unwrap();
        fs::write(tmpdir.path().join("visible.rs"), "fn visible() {}").unwrap();
        // This markdown file should NOT appear because it isn't in .agentsee
        fs::write(tmpdir.path().join("README.md"), "# not listed").unwrap();
        // This excluded .py should NOT appear because excluded/ is force-excluded
        fs::write(tmpdir.path().join("excluded/keep.py"), "print('nope')").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                // Included by pattern
                assert!(
                    content.contains("visible.rs"),
                    "visible.rs should be included; content: {content}"
                );
                assert!(
                    content.contains("secret.py"),
                    "secret.py should be included; content: {content}"
                );
                // Excluded by ! pattern
                assert!(
                    !content.contains("hidden.rs"),
                    "hidden.rs in excluded/ should be force-excluded; content: {content}"
                );
                assert!(
                    !content.contains("keep.py"),
                    "keep.py in excluded/ should be force-excluded; content: {content}"
                );
                // Not in .agentsee at all
                assert!(
                    !content.contains("README.md"),
                    "README.md should not be included (not in .agentsee); content: {content}"
                );
                // Priority order: *.rs files should appear before *.py files
                let rs_pos = content.find("### `visible.rs`").unwrap();
                let py_pos = content.find("### `secret.py`").unwrap();
                assert!(
                    rs_pos < py_pos,
                    "*.rs (higher priority) must appear before *.py in output"
                );
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_agentsee_budget_skips_low_priority_first() {
        let tmpdir = TempDir::new().unwrap();
        // High priority: *.toml; lower: *.rs
        fs::write(tmpdir.path().join(".agentsee"), "*.toml\n*.rs\n").unwrap();
        // Small high-priority file
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();
        // Large low-priority file that won't fit with Cargo.toml in budget.
        // Cargo.toml is ~28 bytes; make big.rs large enough so combined > MAX.
        let big_content = "x".repeat(MAX_TOTAL_BYTES);
        fs::write(tmpdir.path().join("big.rs"), &big_content).unwrap();
        // Small low-priority file that fits after big.rs is skipped.
        fs::write(tmpdir.path().join("small.rs"), "fn main() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                // Cargo.toml (high priority) should be included
                assert!(content.contains("Cargo.toml"));
                // big.rs is lower priority; would exceed budget, skipped
                assert!(!content.contains("big.rs"));
                // small.rs is also lower priority but fits after big.rs was
                // skipped — however it sorts after big.rs alphabetically and
                // budget may or may not allow it. The key invariant: high
                // priority files come first.
            }
            other => panic!("expected SendMessage action, got: {other:?}"),
        }
    }

    #[test]
    fn inject_no_agentsee_includes_all() {
        // Without .agentsee, behavior is unchanged — all recognized files included.
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(tmpdir.path().join("b.py"), "print('b')").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app);

        match result.action {
            Some(AppAction::SendMessage(content)) => {
                assert!(content.contains("a.rs"));
                assert!(content.contains("b.py"));
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
