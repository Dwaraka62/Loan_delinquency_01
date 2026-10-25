How do I generate a Codex Prompt trail file for every session?

Codex automatically creates a session trail as JSONL files here on Windows:
C:\Users\<your-user>\.codex\sessions\YYYY\MM\DD\rollout-*.jsonl

Each file contains the session’s prompts, responses, tool calls, and events. No extra setting is required.

To find the latest trail:

Get-ChildItem "$env:USERPROFILE\.codex\sessions" -Recurse -Filter "rollout-*.jsonl" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 10 FullName, LastWriteTime

  To copy the latest one with a clearer name:

  $session = Get-ChildItem "$env:USERPROFILE\.codex\sessions" -Recurse -Filter "rollout-*.jsonl" |
  Sort-Object LastWriteTime -Descending |
  Select-Object -First 1

Copy-Item $session.FullName ".\prompt-trail-$($session.BaseName).jsonl"

Be careful when sharing these files: they may contain source code, file contents, credentials accidentally included in prompts, and tool output.

Official reference: Codex configuration schema.

Worked for 45s