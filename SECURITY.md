# Security

mlx-chat is a single-user local tool, not a multi-user service. Security fixes
target the latest release on `main`.

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting](https://github.com/DevZoneX/infer-on-mac/security/advisories/new).
Do not open a public issue with exploit details. Include the version or commit,
macOS and chip, installed extras, reproduction steps, and impact.

## Agent tools

The agent acts with your user account's permissions inside the workspace folder.

- **Confinement.** File tools resolve paths, including symlinks, and refuse
  anything outside the workspace. `run_command` starts in the workspace but is a
  normal shell. It can reach anything your account can, so approve commands with
  the same care as running them yourself.
- **Approvals.** `ask` confirms every edit, command, and web request. `edits`
  applies file edits without asking. `all` skips every prompt. Use `all` only in a
  folder you can restore, such as a clean git checkout. Non-interactive runs
  decline anything that needs approval unless `--approve` allows it.
- **Prompt injection.** File contents, command output, fetched pages, and search
  results go to the model as data, but a model can still be steered by text it
  reads. Review approval cards carefully when the agent has read untrusted content.
- **Web tools.** `fetch_url` accepts only `http` and `https` URLs and always needs
  approval, except in `all` mode.

## Operating boundaries

- The server binds to `127.0.0.1`. `--allow-network` exposes it to other hosts
  without authentication or TLS. Never expose it to the Internet.
- Sessions, history, profiles, exports, and logs are unencrypted local files.
  Sessions can contain attached files and tool output.
- Folder attachments skip hidden and generated entries but do not read
  `.gitignore`. Check the file list before sending private code.
- Extra server arguments (`--mlx-arg`) are passed to MLX-LM unchanged. Review
  options and model repositories that allow remote code execution.
- Shutdown signals only processes whose identity matches a tracked instance.
