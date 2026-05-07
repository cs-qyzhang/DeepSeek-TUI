//! Full-project code injection command: `/inject-full-codes`
//!
//! Walks the workspace directory using the `ignore` crate (respecting
//! `.gitignore`, `.ignore`, `.deepseekignore`) and injects every file
//! matched by the `.agentsee` include filter. Each file's full content
//! is injected into the conversation as a synthetic `read_file` tool-call
//! / tool-result pair so the model sees the files exactly as it would see
//! real tool output — same format, same content-compaction path, no
//! special framing.
//!
//! A `.agentsee` file **must** exist at the workspace root.  It uses
//! gitignore syntax, but inverted: patterns specify what to **include**;
//! `!` patterns specify what to **force-exclude**.  Pattern matching
//! follows gitignore semantics: last matching pattern wins.

use glob::Pattern;
use ignore::WalkBuilder;
use std::path::Path;

use crate::models::{ContentBlock, Message};
use crate::tui::app::{App, AppAction};
use crate::tui::history::HistoryCell;

use super::CommandResult;

/// Default token budget for injected file content (~800 KB ≈ 200K tokens).
#[allow(dead_code)]
pub const DEFAULT_MAX_INJECT_TOKENS: usize = 200_000;

/// Convert a token budget to a byte budget using the ~4 chars/token heuristic.
fn tokens_to_bytes(tokens: usize) -> usize {
    tokens.saturating_mul(4)
}

/// A parsed include rule from a `.agentsee` line.
///
/// Each rule compiles to one or more `glob::Pattern`s that collectively
/// match the same set of paths as the original gitignore pattern would.
struct IncludeRule {
    /// Compiled glob patterns (any match = the rule matches).
    patterns: Vec<Pattern>,
    /// Original pattern text (e.g. "docs/", "*.rs"), used to determine
    /// specificity for priority tie-breaking.
    original: String,
    /// Sequential rule number in the `.agentsee` file (0, 1, 2, …).
    /// Used for both last-match-wins exclusion and priority ordering.
    position: usize,
}

/// Parsed `.agentsee` include filter (gitignore syntax, inverted).
///
/// Each non-blank, non-comment line is either an include pattern or
/// (when prefixed with `!`) an exclude pattern.  Patterns are evaluated
/// in file order; the **last** matching pattern determines whether a
/// path is included or excluded (gitignore semantics).  When multiple
/// include patterns match, the most specific one sets the priority.
struct Agentsee {
    includes: Vec<IncludeRule>,
    /// Exclude rules with their file position for last-match-wins ordering.
    excludes: Vec<(usize, Vec<Pattern>)>,
}

impl Agentsee {
    /// Load `.agentsee` from the workspace root.  Returns `None` when the
    /// file is missing or empty.
    fn load(workspace: &Path) -> Option<Self> {
        let path = workspace.join(".agentsee");
        let content = std::fs::read_to_string(&path).ok()?;
        if content.trim().is_empty() {
            return None;
        }

        let mut includes = Vec::new();
        let mut excludes = Vec::new();
        let mut seq: usize = 0; // sequential position across all rule lines

        for raw in content.lines() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let (negated, pat_str) = if let Some(rest) = line.strip_prefix('!') {
                (true, rest.trim())
            } else {
                (false, line)
            };

            let globs = gitignore_to_glob_patterns(pat_str);
            let patterns: Vec<Pattern> =
                globs.iter().filter_map(|g| Pattern::new(g).ok()).collect();
            if patterns.is_empty() {
                seq += 1;
                continue;
            }

            if negated {
                excludes.push((seq, patterns));
            } else {
                includes.push(IncludeRule {
                    patterns,
                    original: pat_str.to_string(),
                    position: seq,
                });
            }
            seq += 1;
        }

        Some(Self { includes, excludes })
    }

    /// Returns the priority index (0 = highest) for a workspace-relative
    /// path, or `None` if it should be excluded.
    ///
    /// Follows gitignore semantics: all rules (includes and excludes) are
    /// evaluated in file order, and the **last** matching rule wins.  An
    /// excluded path returns `None`.  When multiple include patterns match,
    /// specificity breaks the tie: patterns containing `/` beat bare
    /// filename patterns; among equals, the longer pattern wins.
    fn priority(&self, rel: &str) -> Option<usize> {
        // Collect all matching rules: (position, is_include).
        let mut matches: Vec<(usize, bool)> = Vec::new();

        for rule in &self.includes {
            if rule.patterns.iter().any(|p| p.matches(rel)) {
                matches.push((rule.position, true));
            }
        }
        for (pos, patterns) in &self.excludes {
            if patterns.iter().any(|p| p.matches(rel)) {
                matches.push((*pos, false));
            }
        }

        // Last match by position wins (gitignore semantics).
        let last = matches.iter().max_by_key(|(pos, _)| pos)?;

        if !last.1 {
            return None; // last match was an exclude
        }

        // Priority: find the most specific matching include pattern.
        // Earlier positions win among equal specificity.
        self.includes
            .iter()
            .filter(|r| r.patterns.iter().any(|p| p.matches(rel)))
            .min_by_key(|r| {
                // Lower sort key = higher priority.
                // Directory patterns (containing `/`) sort before
                // bare patterns; among equals, longer is more
                // specific → negate length so longer sorts first.
                let is_dir = if r.original.contains('/') { 0 } else { 1 };
                let neg_len = -(r.original.len() as i64);
                (is_dir, neg_len, r.position)
            })
            .map(|r| r.position)
    }
}

/// Convert a single gitignore-style pattern into one or more standard
/// glob patterns that collectively match the same set of paths.
///
/// Gitignore semantics (as documented in gitignore(5)):
/// - Patterns without `/` match the file **basename** at any depth.
/// - Patterns containing `/` are anchored relative to the `.agentsee`
///   location (workspace root).
/// - A trailing `/` matches only directories (and their contents).
/// - A leading `/` anchors to the workspace root (same effect as having
///   a `/` in the middle for anchoring purposes, but constrains
///   basename-only patterns to root).
/// - `!` negation is handled at the rule level, not here.
fn gitignore_to_glob_patterns(pat: &str) -> Vec<String> {
    let pat = pat.trim();

    // Trailing `/` — matches directories and everything under them.
    if let Some(prefix) = pat.strip_suffix('/') {
        let prefix = prefix.strip_prefix('/').unwrap_or(prefix);
        return vec![format!("{prefix}/**")];
    }

    // Leading `/` — anchored to root (same dir as .agentsee).
    let (anchored, body) = if let Some(rest) = pat.strip_prefix('/') {
        (true, rest)
    } else {
        (false, pat)
    };

    // Contains `/` in the middle — anchored pattern.
    // Matches both the exact path (if a file) and recursively (if a dir).
    if body.contains('/') {
        return vec![body.to_string(), format!("{body}/**")];
    }

    // No `/` — basename match.
    if anchored {
        // `/foo` — match `foo` at root only (file or directory).
        vec![body.to_string(), format!("{body}/**")]
    } else {
        // `*.rs`, `README.md` — match at any depth.
        vec![format!("**/{body}"), format!("**/{body}/**")]
    }
}

/// Result of collecting project files for injection.
struct InjectPlan {
    /// Collected files: (relative_path, content) in priority order.
    files: Vec<(String, String)>,
    /// Number of files included.
    file_count: usize,
    /// Total bytes of file contents.
    total_bytes: usize,
    /// Number of files skipped due to budget.
    skipped_count: usize,
    /// Per-file token estimates: (rel_path, estimated_tokens) in priority order.
    file_tokens: Vec<(String, usize)>,
}

/// Walk the workspace and collect files matched by `.agentsee`.
/// Returns `None` when no `.agentsee` file exists or no files match.
///
/// Files are collected in priority order: files matching earlier patterns
/// are read first.  When budget is tight, later (less important) files are
/// naturally skipped.
fn collect_project_files(workspace: &Path, max_bytes: usize) -> Option<InjectPlan> {
    if !workspace.is_dir() {
        return None;
    }

    let agentsee = Agentsee::load(workspace)?;

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

        let rel = path.strip_prefix(workspace).unwrap_or(path);
        let rel_str = rel.to_string_lossy().replace('\\', "/");

        // .agentsee is the sole authority on what to include.
        let Some(priority) = agentsee.priority(&rel_str) else {
            continue;
        };

        candidates.push((priority, rel_str.to_string(), path.to_path_buf()));
    }

    if candidates.is_empty() {
        return None;
    }

    // Sort by priority (lower first), then by path for deterministic output.
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    // Pass 2: read files in priority order, accumulating up to budget.
    let mut files: Vec<(String, String)> = Vec::new();
    let mut file_tokens: Vec<(String, usize)> = Vec::new();
    let mut total_bytes: usize = 0;
    let mut skipped_count: usize = 0;

    for (_priority, rel_str, abs_path) in &candidates {
        let Ok(content) = std::fs::read_to_string(abs_path) else {
            continue;
        };
        if content.trim().is_empty() {
            continue;
        }

        if total_bytes + content.len() > max_bytes {
            skipped_count += 1;
            continue;
        }

        let content_tokens = content.chars().count().div_ceil(4);
        file_tokens.push((rel_str.clone(), content_tokens));
        total_bytes += content.len();
        files.push((rel_str.clone(), content));
    }

    Some(InjectPlan {
        files,
        file_count: file_tokens.len(),
        total_bytes,
        skipped_count,
        file_tokens,
    })
}

/// Parse the `/inject` or `/fct` command argument for an optional
/// `--max-tokens N` prefix.  Returns `(override_tokens, user_text)`.
fn parse_inject_arg(arg: Option<&str>) -> (Option<usize>, Option<String>) {
    let arg = match arg {
        Some(a) => a,
        None => return (None, None),
    };
    // Try to strip `--max-tokens` prefix (space-separated from the number).
    let rest = match arg.strip_prefix("--max-tokens") {
        Some(r) => r.trim_start(),
        None => return (None, Some(arg.to_string())),
    };
    if rest.is_empty() {
        return (None, None); // incomplete flag, fall back to config default
    }
    // Split off the token count; everything after the first whitespace is user text.
    if let Some((num_str, trailing)) = rest.split_once(char::is_whitespace) {
        let tokens = num_str.parse::<usize>().ok();
        let user_text = if trailing.trim().is_empty() {
            None
        } else {
            Some(trailing.trim().to_string())
        };
        (tokens, user_text)
    } else {
        (rest.parse::<usize>().ok(), None)
    }
}

/// Walk the workspace and inject every project file into the context as
/// synthetic `read_file` tool-call / tool-result pairs.
///
/// Each file becomes:
/// - An assistant message with a `ToolUse` block (id = `inj_N`,
///   name = `read_file`, input = `{"path": "rel/path"}`)
/// - A user message with a `ToolResult` block carrying the file content.
///
/// Each file produces one assistant/user message pair, yielding a
/// sequential read_file(1) → result(1) → read_file(2) → result(2) pattern.
///
/// When `user_text` is present (e.g. from `/inject 总结项目内容`), it is
/// appended as a final user message after all tool results so the model
/// receives both the full project context and the user's request.
pub fn inject_full_codes(app: &mut App, raw_arg: Option<String>) -> CommandResult {
    let (max_tokens_override, user_text) =
        parse_inject_arg(raw_arg.as_deref());
    let max_tokens = max_tokens_override.unwrap_or(app.max_inject_tokens);
    let max_bytes = tokens_to_bytes(max_tokens);

    if !app.workspace.join(".agentsee").exists() {
        return CommandResult::message(
            "No .agentsee file found at workspace root.\n\n\
             Create a .agentsee file to specify which files to inject. \
             Each line is a gitignore-style pattern for files to INCLUDE \
             (e.g. \"*.rs\", \"src/\"). Lines starting with ! are force-excluded \
             (e.g. \"!tests/\").",
        );
    }
    let Some(plan) = collect_project_files(&app.workspace, max_bytes) else {
        return CommandResult::message(
            "No files matched the .agentsee patterns. \
             Check that your patterns match files in the workspace.",
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

    let file_count = plan.file_count;
    let total_kb = plan.total_bytes / 1024;

    // --- Header: system cell in transcript only (not in api_messages) ---
    let header_text = format!(
        "## Full Project Code Injection\n\n\
         Workspace: {}\n\
         Files injected: {} (~{} KB total){}\n\n\
         The following files have been loaded via read_file calls and are \
         now available in your context.",
        app.workspace.display(),
        file_count,
        total_kb,
        skipped_note,
    );
    app.push_history_cell(HistoryCell::System {
        content: header_text.clone(),
    });

    // --- Tool calls: one assistant/user pair per file (sequential) ---
    for (i, (rel_path, content)) in plan.files.iter().enumerate() {
        let tool_id = format!("inj_{i}");

        // Assistant message: the read_file tool call
        app.api_messages.push(Message {
            role: "assistant".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: tool_id.clone(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path": rel_path}),
                caller: None,
            }],
        });

        // User message: the tool result with file content
        app.api_messages.push(Message {
            role: "user".to_string(),
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_id,
                content: content.clone(),
                is_error: None,
                content_blocks: None,
            }],
        });
    }

    // --- History cells for display ---
    // Add a system cell summarizing the tool calls so the user sees what
    // happened in the transcript — tool messages don't produce visible
    // history cells on their own.
    // Wrap each path in backticks so the markdown renderer treats the
    // content as inline code, suppressing bold/italic interpretation of
    // underscores (e.g. __init__.py).
    let mut file_list = String::new();
    for (path, _content) in &plan.files {
        file_list.push_str(&format!("  `{path}`\n"));
    }
    app.push_history_cell(HistoryCell::System {
        content: format!(
            "Injected {} files as read_file calls:\n{file_list}",
            file_count,
        ),
    });

    app.mark_history_updated();

    // --- Optional user text: a final user message that triggers the turn ---
    // Push the user text (or a minimal trigger if none) into api_messages
    // and send it to the engine. The engine must sync api_messages first
    // so it sees the tool calls — the SendMessage handler in the event loop
    // is patched to sync before dispatching.
    let trigger_text = if let Some(text) = user_text.filter(|t| !t.trim().is_empty()) {
        text
    } else {
        // Without user text, send the header as the triggering user message.
        header_text
    };

    CommandResult::with_message_and_action(
        format!(
            "Injected {} files (~{} KB) into context as read_file calls (budget: ~{}K tokens){}",
            file_count, total_kb, max_tokens / 1000, skipped_note
        ),
        AppAction::SendMessage(trigger_text),
    )
}

/// Dry-run the injection and estimate how many tokens the full set of
/// messages would consume. Does NOT modify the conversation.
pub fn full_codes_tokens(app: &App, raw_arg: Option<String>) -> CommandResult {
    let (max_tokens_override, _user_text) =
        parse_inject_arg(raw_arg.as_deref());
    let max_tokens = max_tokens_override.unwrap_or(app.max_inject_tokens);
    let max_bytes = tokens_to_bytes(max_tokens);

    if !app.workspace.join(".agentsee").exists() {
        return CommandResult::message(
            "No .agentsee file found at workspace root. \
             Create one first, then use /full-codes-tokens to estimate.",
        );
    }
    let Some(plan) = collect_project_files(&app.workspace, max_bytes) else {
        return CommandResult::message(
            "No files matched the .agentsee patterns.",
        );
    };

    let kb = plan.total_bytes / 1024;

    // Estimate tokens for the header message + tool calls + tool results.
    let header_char_count = 256; // rough estimate for the header text
    let mut total_chars = header_char_count;
    // Each tool call: ~"read_file" + path + JSON framing ≈ path.len() + 64
    // Each tool result: file content + tool_use_id ≈ content.len() + 32
    for (path, tokens) in &plan.file_tokens {
        total_chars += path.len() + 64; // tool use
        total_chars += tokens.saturating_mul(4) + 32; // tool result (tokens→chars approximation)
    }
    let token_estimate = total_chars.div_ceil(4);

    let skipped_line = if plan.skipped_count > 0 {
        format!("\nFiles skipped (budget): {}", plan.skipped_count)
    } else {
        String::new()
    };

    // Build per-file breakdown in priority order.
    // +2 for the backtick wrappers around each path.
    let max_path_len = plan
        .file_tokens
        .iter()
        .map(|(p, _)| p.len() + 2)
        .max()
        .unwrap_or(0)
        .max(8);
    let mut file_list = String::new();
    let mut content_sum: usize = 0;
    for (path, tokens) in &plan.file_tokens {
        content_sum += tokens;
        let padding = " ".repeat(max_path_len.saturating_sub(path.len() + 2) + 2);
        file_list.push_str(&format!("  `{path}`{padding}~{tokens} tokens\n"));
    }
    let framing_tokens = token_estimate.saturating_sub(content_sum);

    CommandResult::message(format!(
        "Full Codes Token Estimate (as read_file tool calls)\n\
         Workspace: {}\n\
         Budget: ~{} tokens\n\
         Files: {}\n\
         Content size: ~{} KB\n\
         Estimated tokens: ~{}  (~4 chars/token heuristic)\n\
           content: ~{}\n\
           framing: ~{}{}\n\
         \n\
         Per-file breakdown (priority order):\n\
         {}",
        app.workspace.display(),
        max_tokens,
        plan.file_count,
        kb,
        token_estimate,
        content_sum,
        framing_tokens,
        skipped_line,
        file_list,
    ))
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

    /// Helper: extract all file paths mentioned in ToolUse blocks in api_messages.
    fn injected_paths(app: &App) -> Vec<String> {
        let mut paths = Vec::new();
        for msg in &app.api_messages {
            for block in &msg.content {
                if let ContentBlock::ToolUse { name, input, .. } = block
                    && name == "read_file"
                    && let Some(path) = input.get("path").and_then(|v| v.as_str())
                {
                    paths.push(path.to_string());
                }
            }
        }
        paths
    }

    /// Helper: extract all file contents from ToolResult blocks in api_messages.
    fn injected_contents(app: &App) -> Vec<String> {
        let mut contents = Vec::new();
        for msg in &app.api_messages {
            for block in &msg.content {
                if let ContentBlock::ToolResult { content, .. } = block {
                    contents.push(content.clone());
                }
            }
        }
        contents
    }

    #[test]
    fn inject_empty_workspace_returns_message() {
        let tmpdir = TempDir::new().unwrap();
        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app, None);
        assert!(result.message.is_some());
        let msg = result.message.unwrap();
        assert!(
            msg.contains("No .agentsee file found"),
            "unexpected message: {msg}"
        );
    }

    #[test]
    fn inject_collects_source_and_doc_files() {
        let tmpdir = TempDir::new().unwrap();
        // .agentsee is now required — patterns define what to include.
        fs::write(tmpdir.path().join(".agentsee"), "*.rs\n*.md\n*.toml\n").unwrap();
        fs::write(tmpdir.path().join("main.rs"), "fn main() {}").unwrap();
        fs::write(tmpdir.path().join("README.md"), "# My Project").unwrap();
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app, None);

        assert!(result.message.is_some());
        let status = result.message.unwrap();
        assert!(status.contains("Injected 3 files"), "got: {status}");

        let paths = injected_paths(&app);
        assert!(paths.contains(&"main.rs".to_string()), "paths: {paths:?}");
        assert!(paths.contains(&"README.md".to_string()), "paths: {paths:?}");
        assert!(paths.contains(&"Cargo.toml".to_string()), "paths: {paths:?}");

        let contents = injected_contents(&app);
        let all_content = contents.join("\n");
        assert!(all_content.contains("fn main() {}"));
        assert!(all_content.contains("# My Project"));
        assert!(all_content.contains("[package]"));

        assert!(
            matches!(&result.action, Some(AppAction::SendMessage(t)) if t.contains("Full Project Code Injection")),
            "action: {:?}",
            result.action
        );
    }

    #[test]
    fn inject_respects_agentsee() {
        let tmpdir = TempDir::new().unwrap();
        // *.rs first (higher priority), *.py second, !excluded/ excludes.
        // Gitignore last-match-wins: !excluded/ at pos 2 overrides earlier
        // include patterns for paths under excluded/.
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
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        // Included by pattern
        assert!(
            paths.contains(&"visible.rs".to_string()),
            "visible.rs should be included; paths: {paths:?}"
        );
        assert!(
            paths.contains(&"secret.py".to_string()),
            "secret.py should be included; paths: {paths:?}"
        );
        // Excluded by ! pattern (last-match: !excluded/ at pos 2 wins)
        assert!(
            !paths.contains(&"excluded/hidden.rs".to_string()),
            "excluded/hidden.rs should be force-excluded; paths: {paths:?}"
        );
        assert!(
            !paths.contains(&"excluded/keep.py".to_string()),
            "excluded/keep.py should be force-excluded; paths: {paths:?}"
        );
        // Not in .agentsee at all
        assert!(
            !paths.contains(&"README.md".to_string()),
            "README.md should not be included (not in .agentsee); paths: {paths:?}"
        );

        // Priority order: *.rs files should appear before *.py files
        let rs_idx = paths
            .iter()
            .position(|p| p == "visible.rs")
            .expect("visible.rs should be present");
        let py_idx = paths
            .iter()
            .position(|p| p == "secret.py")
            .expect("secret.py should be present");
        assert!(
            rs_idx < py_idx,
            "*.rs (higher priority) must appear before *.py in tool call list"
        );
    }

    /// When `!exclude` comes *before* an include pattern in the file,
    /// gitignore last-match-wins means the later include overrides the
    /// earlier exclude — a file matching both will be included.
    #[test]
    fn inject_agentsee_last_match_wins_include_overrides_early_exclude() {
        let tmpdir = TempDir::new().unwrap();
        // !excluded/ at pos 0, *.py at pos 1.
        // A .py file in excluded/ matches both; *.py (pos 1) wins → included.
        fs::write(
            tmpdir.path().join(".agentsee"),
            "!excluded/\n*.py\n",
        )
        .unwrap();
        fs::create_dir(tmpdir.path().join("excluded")).unwrap();
        fs::write(tmpdir.path().join("excluded/keep.py"), "print('kept')").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        assert!(
            paths.contains(&"excluded/keep.py".to_string()),
            "excluded/keep.py should be included: *.py (pos 1) overrides !excluded/ (pos 0); paths: {paths:?}"
        );
    }

    #[test]
    fn inject_agentsee_budget_skips_low_priority_first() {
        let tmpdir = TempDir::new().unwrap();
        // High priority: *.toml; lower: *.rs
        fs::write(tmpdir.path().join(".agentsee"), "*.toml\n*.rs\n").unwrap();
        // Small high-priority file
        fs::write(tmpdir.path().join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();
        // Large low-priority file that won't fit with Cargo.toml in budget.
        let big_content = "x".repeat(tokens_to_bytes(DEFAULT_MAX_INJECT_TOKENS));
        fs::write(tmpdir.path().join("big.rs"), &big_content).unwrap();
        // Small low-priority file that fits after big.rs is skipped.
        fs::write(tmpdir.path().join("small.rs"), "fn main() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        // Cargo.toml (high priority) should be included
        assert!(paths.contains(&"Cargo.toml".to_string()));
        // big.rs is lower priority; would exceed budget, skipped
        assert!(!paths.contains(&"big.rs".to_string()));
    }

    #[test]
    fn inject_agentsee_nested_dir_priority() {
        // Patterns with trailing `/` match directory contents.
        // `core/utils/` (pos 0) has higher priority than `core/` (pos 1).
        let tmpdir = TempDir::new().unwrap();
        fs::write(
            tmpdir.path().join(".agentsee"),
            "core/utils/\ncore/\n",
        )
        .unwrap();
        fs::create_dir_all(tmpdir.path().join("core/utils")).unwrap();
        fs::write(
            tmpdir.path().join("core/utils/helpers.py"),
            "def help(): pass",
        )
        .unwrap();
        fs::write(tmpdir.path().join("core/lib.rs"), "pub mod utils;").unwrap();
        // This file should NOT match any pattern
        fs::write(tmpdir.path().join("README.md"), "# readme").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        // core/utils/ has higher priority → helpers.py first
        let utils_idx = paths
            .iter()
            .position(|p| p.contains("helpers.py"))
            .expect("helpers.py should be present");
        let lib_idx = paths
            .iter()
            .position(|p| p == "core/lib.rs")
            .expect("core/lib.rs should be present");
        assert!(
            utils_idx < lib_idx,
            "core/utils/ (higher priority) must come before core/ files"
        );
        assert!(!paths.contains(&"README.md".to_string()));
    }

    #[test]
    fn inject_agentsee_specificity_wins_over_bare_pattern() {
        let tmpdir = TempDir::new().unwrap();
        // README.md (pos 0, basename-only) vs docs/ (pos 1, directory).
        // docs/README.md matches both, but docs/ is more specific (has `/`),
        // so it gets docs/'s priority (pos 1). Root README.md only matches
        // the bare pattern (pos 0).
        fs::write(tmpdir.path().join(".agentsee"), "README.md\ndocs/\n").unwrap();
        fs::create_dir_all(tmpdir.path().join("docs")).unwrap();
        fs::write(tmpdir.path().join("docs/README.md"), "# sub readme").unwrap();
        fs::write(tmpdir.path().join("README.md"), "# root readme").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        assert!(paths.contains(&"README.md".to_string()));
        assert!(paths.contains(&"docs/README.md".to_string()));
        // Root README.md (priority from pos 0) comes before docs/ (pos 1).
        let root_pos = paths
            .iter()
            .position(|p| p == "README.md")
            .expect("root README.md should be present");
        let docs_pos = paths
            .iter()
            .position(|p| p == "docs/README.md")
            .expect("docs/README.md should be present");
        assert!(
            root_pos < docs_pos,
            "root README.md (pos 0) should come before docs/README.md (pos 1)"
        );
    }

    #[test]
    fn inject_no_agentsee_reports_missing_file() {
        // Without .agentsee, the command now returns an error telling the
        // user to create one.
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join("a.rs"), "fn a() {}").unwrap();
        fs::write(tmpdir.path().join("b.py"), "print('b')").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app, None);

        let msg = result.message.unwrap();
        assert!(
            msg.contains("No .agentsee file found"),
            "expected missing-.agentsee error, got: {msg}"
        );
    }

    #[test]
    fn inject_respects_gitignore() {
        let tmpdir = TempDir::new().unwrap();
        // .agentsee required; `.ignore` excludes ignored/ dir.
        fs::write(tmpdir.path().join(".agentsee"), "*.rs\n").unwrap();
        fs::write(tmpdir.path().join(".ignore"), "ignored/\n").unwrap();
        fs::create_dir(tmpdir.path().join("ignored")).unwrap();
        fs::write(tmpdir.path().join("ignored/secret.rs"), "fn secret() {}").unwrap();
        fs::write(tmpdir.path().join("visible.rs"), "fn visible() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        assert!(paths.contains(&"visible.rs".to_string()), "paths: {paths:?}");
        assert!(
            !paths.contains(&"secret.rs".to_string()),
            "ignored file leaked: {paths:?}"
        );
    }

    #[test]
    fn inject_skips_empty_files() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join(".agentsee"), "*.rs\n").unwrap();
        fs::write(tmpdir.path().join("empty.rs"), "").unwrap();
        fs::write(tmpdir.path().join("real.rs"), "fn real() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        assert!(paths.contains(&"real.rs".to_string()));
        assert!(!paths.contains(&"empty.rs".to_string()));
    }

    #[test]
    fn inject_skips_files_not_in_agentsee() {
        // Files with any extension are skipped unless .agentsee matches them.
        // There is no built-in extension whitelist.
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join(".agentsee"), "*.rs\n").unwrap();
        fs::write(tmpdir.path().join("image.png"), "fake-png-data").unwrap();
        fs::write(tmpdir.path().join("lib.rs"), "pub fn x() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let _ = inject_full_codes(&mut app, None);

        let paths = injected_paths(&app);
        assert!(paths.contains(&"lib.rs".to_string()));
        assert!(!paths.contains(&"image.png".to_string()));
    }

    #[test]
    fn inject_with_user_text_adds_final_message() {
        let tmpdir = TempDir::new().unwrap();
        fs::write(tmpdir.path().join(".agentsee"), "*.rs\n").unwrap();
        fs::write(tmpdir.path().join("main.rs"), "fn main() {}").unwrap();

        let mut app = create_test_app_in(&tmpdir);
        let result = inject_full_codes(&mut app, Some("总结项目内容".to_string()));

        assert!(result.message.is_some());
        let status = result.message.unwrap();
        assert!(status.contains("Injected 1 file"), "got: {status}");

        // The trigger for the next turn is the user's text
        assert!(
            matches!(&result.action, Some(AppAction::SendMessage(t)) if t == "总结项目内容"),
            "action: {:?}",
            result.action
        );

        let paths = injected_paths(&app);
        assert!(paths.contains(&"main.rs".to_string()));
    }

    // --- gitignore_to_glob_patterns conversion tests ---

    #[test]
    fn glob_patterns_basename_match() {
        // Patterns without `/` match the basename at any depth.
        let pats = gitignore_to_glob_patterns("*.rs");
        let patterns: Vec<Pattern> = pats.iter().filter_map(|g| Pattern::new(g).ok()).collect();
        assert!(patterns.iter().any(|p| p.matches("main.rs")));
        assert!(patterns.iter().any(|p| p.matches("src/main.rs")));
        assert!(patterns.iter().any(|p| p.matches("deeply/nested/path/mod.rs")));
    }

    #[test]
    fn glob_patterns_bare_name_matches_any_depth() {
        let pats = gitignore_to_glob_patterns("README.md");
        let patterns: Vec<Pattern> = pats.iter().filter_map(|g| Pattern::new(g).ok()).collect();
        assert!(patterns.iter().any(|p| p.matches("README.md")));
        assert!(patterns.iter().any(|p| p.matches("docs/README.md")));
    }

    #[test]
    fn glob_patterns_anchored_by_slash() {
        // Patterns with `/` are anchored to the workspace root.
        // `docs/` → `docs/**` matches only files under docs/ at root.
        let pats = gitignore_to_glob_patterns("docs/");
        let patterns: Vec<Pattern> = pats.iter().filter_map(|g| Pattern::new(g).ok()).collect();
        assert!(patterns.iter().any(|p| p.matches("docs/README.md")));
        assert!(patterns.iter().any(|p| p.matches("docs/sub/file.md")));
        assert!(!patterns.iter().any(|p| p.matches("other/docs/README.md")));
        assert!(!patterns.iter().any(|p| p.matches("README.md")));
    }

    #[test]
    fn glob_patterns_leading_slash_anchors_basename() {
        // `/README.md` matches README.md at root only (not in subdirs).
        let pats = gitignore_to_glob_patterns("/README.md");
        let patterns: Vec<Pattern> = pats.iter().filter_map(|g| Pattern::new(g).ok()).collect();
        assert!(patterns.iter().any(|p| p.matches("README.md")));
        assert!(!patterns.iter().any(|p| p.matches("docs/README.md")));
        assert!(!patterns.iter().any(|p| p.matches("src/README.md")));
    }

    #[test]
    fn glob_patterns_middle_slash_anchored() {
        // `foo/bar` matches `foo/bar` at root, not `x/foo/bar`.
        let pats = gitignore_to_glob_patterns("foo/bar");
        let patterns: Vec<Pattern> = pats.iter().filter_map(|g| Pattern::new(g).ok()).collect();
        assert!(patterns.iter().any(|p| p.matches("foo/bar")));
        assert!(patterns.iter().any(|p| p.matches("foo/bar/baz.txt")));
        assert!(!patterns.iter().any(|p| p.matches("x/foo/bar")));
        assert!(!patterns.iter().any(|p| p.matches("x/foo/bar/baz.txt")));
    }
}
