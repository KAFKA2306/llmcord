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
  └─ deterministic watchdog
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
- watchdog state machine
- restart 回数上限と cooldown
- restart 後に probe が成功するまで対象 Local LLM の受付を再開しない
- optional NVIDIA GPU / VRAM residency check
- production manifest の fail-closed validation

## セットアップ

必要なもの:

- uv
- Discord Bot Token
- Discord Developer Portal で有効化した `MESSAGE CONTENT INTENT`
- OpenAI-compatible LLM endpoint

開発・非production実行:

```bash
git clone https://github.com/KAFKA2306/llmcord.git
cd llmcord
uv sync --locked
export DISCORD_BOT_TOKEN='...'
uv run python llmcord.py
```

production実行では `production_entrypoint.py` を使用します。

```bash
uv run --locked --no-sync python production_entrypoint.py
```

Docker image もこの entrypoint を使用します。

```bash
docker compose up --build
```

依存関係の正本は `pyproject.toml`、固定解は `uv.lock` です。

## Production contract

production の backend / model / network / supervisor / GPU 条件は `config.yaml` の `production` を唯一の deployment authority とします。

#7 の実機比較が終わるまでは `production.enabled: false` のままにし、backend や model を推測で固定しません。採用対象が決まった後に manifest を埋めて `true` にします。

```yaml
production:
  enabled: true
  backend: llamacpp
  backend_version_or_commit: <pinned release or commit>
  model: llamacpp/<model-alias>
  model_artifact:
    upstream: <exact upstream model>
    artifact: <exact artifact filename>
    revision: <pinned revision or commit>
    sha256: <64-character sha256>
    quantization_or_dtype: <exact quantization/dtype>
    context_window_tokens: <actual production context>
    # verify_path: /models/model.gguf
  network:
    mode: native
    endpoint: http://127.0.0.1:8080/v1
  supervisor:
    kind: systemd
    restart_command: ["systemctl", "restart", "llama-server.service"]
  gpu:
    required: true
    device_index: 0
    min_vram_used_mib: <measured production floor>
```

`production_entrypoint.py` は Bot 起動前に `production_contract.py` で検証します。production有効時は以下を許可しません。

- `latest` / `main` / `master` 等の floating runtime/model revision
- model SHA-256 の欠落・形式不正
- production model と `models` / watchdog対象の不一致
- production endpoint と `providers.<backend>.base_url` の不一致
- Docker mode で `localhost` / `127.0.0.1` を backend endpoint として使用
- production supervisor と `health_control.restart_command` の不一致
- GPU必須なのに watchdog GPU check が無効
- production model が起動時選択modelと一致しない構成

`model_artifact.verify_path` を指定し、実行環境からmodel fileが見える場合は startup 時に SHA-256 も実測照合します。hash が違えば Discord Bot を起動しません。

既存 `providers` / `models` / `health_control` は実行用設定として残りますが、production有効時は manifest と一致していることを必須化します。値が食い違ったまま片方だけ更新して運用することはできません。

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

production の `health_control` は `production` manifest と一致させます。状態は `starting → healthy → suspect → degraded → recovering` と遷移し、連続失敗が threshold に達した時だけ対象 model の新規受付を停止します。restart は argv として直接実行し shell を使いません。restart 上限を超えて無限 restart しません。

restart 後は、設定した GPU check と synthetic generation probe の両方が成功して初めて `healthy` に戻ります。

## 検証

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile \
  llmcord.py backend_probe.py context_management.py runtime_control.py health_control.py \
  production_contract.py production_entrypoint.py
uv run --locked --no-sync python -m unittest discover -s tests -v
docker build -t llmcord:test .
```

CI / mock test の成功だけを production 成功とは扱いません。実 Discord・実 Local LLM・実 GPU での startup / restart / rollback / process kill / generation hang / GPU failure / repeated compaction / 24h soak は #13 / #14 の実機受入で検証します。

## 主なファイル

```text
llmcord.py                  Discord bridge
context_management.py       token-aware auto compaction
runtime_control.py           queue / concurrency / generation deadlines
backend_probe.py             real-generation health probe
health_control.py            watchdog / restart / optional GPU check
production_contract.py       canonical deployment validation
production_entrypoint.py     production fail-closed startup path
config.yaml                  runtime + canonical production manifest
pyproject.toml               dependency authority
uv.lock                      locked dependencies
```

## License

MIT License。`LICENSE.md` を参照してください。
