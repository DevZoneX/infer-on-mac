# Security

Security fixes target the current `main` branch. This is a local inference tool,
not an authenticated multi-user service.

## Reporting

Use GitHub's private vulnerability reporting if it is enabled for this repository.
If it is unavailable, open an issue requesting a private contact method without
including exploit details, credentials, private prompts, or source files.

Include the affected commit, platform, installation extras, reproduction steps,
and the impact. Keep sensitive values out of public logs and screenshots.

## Operating boundaries

- The default endpoint is loopback. `--allow-network` permits other hosts, but
  the app does not add authentication or TLS. Do not expose it to the Internet.
- Normal inference/audio processing is local. Model and phonemizer downloads,
  search-provider requests, and explicitly selected remote endpoints use networks.
- Sessions, history, profiles, exports, and logs are ordinary local files. They
  are not encrypted. Sessions/JSON exports can include attached file excerpts.
- Folder attachments skip hidden/generated entries but do not interpret
  `.gitignore`. Inspect the selected list before sending private source context.
- Attached text and search snippets are reference data. The app does not execute
  their contents or automatically apply edits.
- Extra MLX server arguments are passed to the selected backend. Review model
  repository code and any options that permit remote code execution.
- Shutdown controls only tracked instances and rechecks process identity. Do not
  bypass those guards to stop an endpoint whose ownership cannot be established.

Keep app data and model caches outside the source checkout. Review exported chats,
logs, recordings, screenshots, and Git history before publishing them.
