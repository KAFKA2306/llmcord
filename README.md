# llmcord

Discord を OpenAI `/v1/chat/completions` 互換 LLM のフロントエンドとして使う Bot です。

元実装: https://github.com/jakobdylanc/llmcord

production 安定化の正本: https://github.com/KAFKA2306/llmcord/issues/1

## 現在の構成

```text
Discord
  ↓
llmcord
  ├─ token-aware auto compaction
  ├─ bounded queue / concurrency
  ├─ finite generation deadlines
  └─ optional watchdog
         ↓
OpenAI-compatible Local LLM
         ↓
NVIDIA GPU
```

会話履歴の正本は Discord の返信チェーンです。通常の会話用 DB は持ちません。

## 実装済みの運用保護

- token-aware automatic context compaction
- system/developer authority と直近会話の原文保持
- oversized text の chunk / compact
- context 超過時の silent truncation 禁止
- `max_concurrency: 1` を既定とする bounded inference admission
- bounded queue と有限 queue wait
- connect / first-generation-signal / stream-idle / total-generation timeout
- OpenAI Python SDK の automatic retry 無効化 (`max_retries=0`)
- Discord message ID を request ID として backend header / log へ伝播
- `/health` ではなく実 `/v1/chat/completions` を使う synthetic generation probe
- optional watchdog state machine
- restart 回数上限と cooldown
- restart 後に probe が成功するまで対象 Local LLM の受付を再開しない
- optional NVIDIA GPU / VRAM residency check

`health_control` は汎用リポジトリでは既定 `false` です。production backend / model / supervisor / GPU 条件を実環境から確定する前に推測値で有効化しません。

## セットアップ

必要なもの:

- uv
- Discord Bot Token
- Discord Developer Portal で有効化した `MESSAGE CONTENT INTENT`
- OpenAI-compatible LLM endpoint

```bash
git clone https://github.com/KAFKA2306/llmcord.git
cd llmcord
uv sync --locked
export DISCORD_BOT_TOKEN='...'
uv run python llmcord.py
```

Docker Compose:

```bash
docker compose up --build
```

依存関係の正本は `pyproject.toml`、固定解は `uv.lock` です。

## Runtime control

```yaml
runtime_control:
  max_concurrency: 1
  max_queue_size: 4
  queue_wait_timeout_seconds: 300
  connect_timeout_seconds: 10
  first_token_timeout_seconds: 300
  stream_idle_timeout_seconds: 120
  total_generation_timeout_seconds: 1800
```

queue 上限を超えた request は backend へ送りません。stream が開始後に停止した場合も `stream_idle_timeout_seconds` で有限時間内に失敗へ移します。

## 自動コンパクション

利用者に「会話を短くする」「新しいスレッドを作る」ことを要求しません。対応 profile では backend と同じ token counting path で active context を測定し、hard limit の手前で古い履歴を自動圧縮します。

```text
system/developer authority
+ compacted older state
+ recent verbatim messages
+ current user input
+ reserved output budget
```

llama-server の例:

```yaml
providers:
  llamacpp:
    base_url: http://localhost:8080/v1

models:
  llamacpp/my-model:
    context_management:
      context_window_tokens: auto
      max_output_tokens: 2048
      safety_margin_tokens: 512
      compaction_trigger_tokens: 12000
      compaction_target_tokens: 8000
      recent_messages: 3
      compaction_max_output_tokens: 1024
```

数値は形式例です。production 値は実 model/runtime から決めます。`context_window_tokens: auto` は llama-server `/props` の `n_ctx`、入力 token 数は `/v1/chat/completions/input_tokens` を使います。取得に失敗した場合は文字数推定へ silent fallback しません。

## Synthetic probe / watchdog

正常性 authority は実生成です。

```text
POST /v1/chat/completions
"Reply exactly: PONG"
```

`backend_probe.py` が HTTP / timeout / protocol / unexpected output を分類します。process、port、`/health`、`/v1/models` が生存しているだけでは healthy としません。

production で watchdog を使う場合:

```yaml
health_control:
  enabled: true
  model: llamacpp/my-model
  probe_interval_seconds: 60
  probe_timeout_seconds: 30
  probe_prompt: "Reply exactly: PONG"
  probe_expected_text: PONG
  probe_max_tokens: 8
  failure_threshold: 2
  restart_cooldown_seconds: 60
  post_restart_grace_seconds: 10
  max_restart_attempts: 3
  restart_command_timeout_seconds: 30
  restart_command: ["systemctl", "restart", "llama-server.service"]
  nvidia_gpu:
    enabled: true
    device_index: 0
    min_vram_used_mib: 12000
    timeout_seconds: 5
```

上記の service 名・VRAM 値は例です。実環境で確認した値へ置き換えます。

状態は `starting → healthy → suspect → degraded → recovering` と遷移します。連続失敗が `failure_threshold` に達した時だけ対象 model の新規受付を停止します。restart は argv として直接実行し shell を使いません。`max_restart_attempts` を超えて無限 restart しません。

restart 後は、設定した GPU check と synthetic generation probe の両方が成功して初めて `healthy` に戻ります。

Docker から host の systemd 等を直接操作できるとは仮定しません。`restart_command` は **llmcord が実際に動く環境から実行可能な supervisor interface** を明示してください。production supervisor の最終 authority は Issue #13 で固定します。

## 検証

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile \
  llmcord.py backend_probe.py context_management.py runtime_control.py health_control.py
uv run --locked --no-sync python -m unittest discover -s tests -v
docker build -t llmcord:test .
```

CI / mock test の成功だけを production 成功とは扱いません。実 Discord・実 Local LLM・実 GPU での process kill / generation hang / GPU failure / repeated compaction / 24h soak は Issue #14 の受入試験で検証します。

## 主なファイル

```text
llmcord.py               Discord bridge
context_management.py    token-aware auto compaction
runtime_control.py        queue / concurrency / generation deadlines
backend_probe.py          real-generation health probe
health_control.py         watchdog / restart / optional GPU check
config.yaml               runtime / provider / model policy
pyproject.toml            dependency authority
uv.lock                   locked dependencies
```

## License

MIT License。`LICENSE.md` を参照してください。
