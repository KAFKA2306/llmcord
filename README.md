# llmcord

Discord を OpenAI `/v1/chat/completions` 互換 LLM のフロントエンドとして使う Bot です。ローカル LLM とリモート LLM の両方を扱えます。

元実装: https://github.com/jakobdylanc/llmcord

長時間の無人運用に必要な残作業は https://github.com/KAFKA2306/llmcord/issues/1 で管理しています。

## 現在の構成

```text
Discord
  ↓
llmcord.py
  ↓ bounded admission / finite deadlines
OpenAI /v1/chat/completions
  ↓
LLM API
  ├─ llama.cpp / llama-server
  ├─ Ollama
  ├─ LM Studio
  ├─ vLLM
  └─ その他の OpenAI 互換 API
```

会話履歴の正本は Discord の返信チェーンです。通常の会話用データベースは持ちません。

## 主な機能

- Discord のメンション、返信、DM、スレッドから会話
- 返信チェーンから会話履歴を再構築
- `/model` による管理者のモデル切り替え
- OpenAI `/v1/chat/completions` 互換 API
- ストリーミング応答と Discord の長文分割
- テキスト・画像添付
- user / role / channel 単位のアクセス制御
- token-aware automatic context compaction
- bounded inference admission / queue
- connect / first-token / total-generation deadline
- OpenAI SDK automatic retry disabled (`max_retries=0`)
- Discord message ID を request ID として backend header / log へ伝播
- `uv` による Python・依存関係管理

## セットアップ

必要なもの:

- uv
- Discord Bot Token
- Discord Developer Portal で有効化した `MESSAGE CONTENT INTENT`
- 使用する LLM の OpenAI 互換 API endpoint

```bash
git clone https://github.com/KAFKA2306/llmcord.git
cd llmcord
uv sync --locked
export DISCORD_BOT_TOKEN='...'
uv run python llmcord.py
```

Client ID も使う場合:

```bash
export DISCORD_CLIENT_ID='...'
```

Docker Compose:

```bash
docker compose up --build
```

依存関係の正本は `pyproject.toml`、固定解は `uv.lock` です。`pip install` や `requirements.txt` は使用しません。

## Runtime control

`runtime_control` はプロセス起動時に読み込み、Local LLM へ送る処理を有限化します。

```yaml
runtime_control:
  max_concurrency: 1
  max_queue_size: 4
  queue_wait_timeout_seconds: 300
  connect_timeout_seconds: 10
  first_token_timeout_seconds: 300
  total_generation_timeout_seconds: 1800
```

`max_concurrency: 1` では、context compaction を含む LLM 処理を同時に1件だけ実行します。待機要求も `max_queue_size` を超えて保持しません。queue が満杯なら request を backend へ送らず Discord へ overload を明示します。queue 待機自体にも有限 timeout があります。

ストリームは次の2段階で deadline を持ちます。

- `first_token_timeout_seconds`: HTTP request 開始から最初の実 generation signal まで
- `total_generation_timeout_seconds`: request 開始から stream 完了まで

空のprotocol chunkだけでは first token 成功扱いにしません。timeout / cancellation / error 時は stream を close します。

OpenAI Python SDK の automatic retry は `max_retries=0` とし、Discord handler / SDK / watchdog が独立に再試行する retry storm を作りません。現時点の production policy は自動再試行なしです。

`runtime_control` は context window の推測値ではなく運用上の admission / deadline policy です。値を変更する場合はプロセスを再起動します。

## LLM 設定

`config.yaml` の `providers` と `models` を使います。

```yaml
providers:
  llamacpp:
    base_url: http://localhost:8080/v1

models:
  llamacpp/my-model:
```

モデル名は `<provider>/<backend が受け付ける model name>` です。

## 自動コンパクション

長い会話を利用者に管理させません。対応する Local LLM profile では、毎回 backend と同じ token counting path で入力 token 数を測り、hard limit に到達する前に古い履歴を自動圧縮します。

active context は次で構成します。

```text
system/developer authority
+ compacted older state
+ recent verbatim messages
+ current user input
+ reserved output budget
```

system/developer message は圧縮しません。Discord の raw reply-chain history も破壊しません。compaction summary は active context 用の派生データです。

通常の context 超過で「会話を短くしてください」「新しいスレッドを作ってください」と利用者へ要求しません。現在入力自体が大きい場合も、テキストであれば chunk → compact を自動実行します。安全に圧縮できない非テキスト入力は黙って削除せず明示的に失敗します。

### llama-server profile

llama-server では以下を使います。

- runtime context window: `GET /props` → `default_generation_settings.n_ctx`
- exact input token count: `POST /v1/chat/completions/input_tokens`
- compaction: 同じ production model の `/v1/chat/completions`

例:

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

上の数値は設定形式の例です。production では実際に起動した model / runtime の context contract に合わせます。`context_window_tokens: auto` は llama-server の実 `n_ctx` を読みます。

`compaction_trigger_tokens` は hard limit より小さく、`compaction_target_tokens` は trigger より小さくする必要があります。hard input limit は次です。

```text
context_window_tokens - max_output_tokens - safety_margin_tokens
```

context management を有効にした場合、`max_text` / `max_messages` は context-window authority ではありません。返信チェーンを token-aware に処理します。古い実装との互換用 guardrail として設定自体は残しています。

`max_tokens` / `max_completion_tokens` を provider/model の別設定へ重複して書くことは禁止しています。出力 token 予算の正本は `context_management.max_output_tokens` です。

backend が `/props` や llama-server の token count endpoint と異なる場合は `props_url` / `token_count_url` を明示できます。取得・計測に失敗した場合は推測値へ silent fallback せず、その request を失敗させます。

コンパクションが発生した場合は Discord 応答へ次を表示します。

```text
ℹ️ Long conversation history was automatically compacted
```

## Discord 入力 guardrail

`config.yaml` の既存設定:

| 設定 | 内容 |
| --- | --- |
| `max_text` | context management 無効時の1メッセージ文字数上限 |
| `max_messages` | context management 無効時の返信チェーン件数上限 |
| `max_images` | 1メッセージから渡す画像数上限 |
| `allow_dms` | DM の許可 |
| `permissions` | user / role / channel の allow / block |

画像数超過や未対応添付は既存 warning で利用者へ明示します。

## 現在まだ Issue #1 に残るもの

- synthetic generation probe
- deterministic watchdog / backend restart
- restart storm protection
- GPU unavailable / CPU fallback detection
- machine-readable metrics
- 実 Local LLM を使った障害注入・soak test

CI や process health だけを production 成功とは扱いません。

## 検証

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile llmcord.py context_management.py runtime_control.py
uv run --locked --no-sync python -m unittest discover -s tests -v
docker build -t llmcord:test .
```

## ファイル

```text
llmcord.py                        Discord Bot 本体
context_management.py             token-aware automatic compaction
runtime_control.py                 bounded admission / finite stream deadlines
config.yaml                        実行設定 / provider / model 定義
pyproject.toml                     Python / 直接依存の正本
uv.lock                            固定済み依存関係
.python-version                    Python 系列
Dockerfile                         uv ベースのコンテナ
docker-compose.yaml                Compose 実行設定
.github/workflows/ci.yml           CI
tests/test_context_management.py   context management tests
tests/test_runtime_control.py       runtime control tests
LICENSE.md                         MIT License
```

## License

MIT License。`LICENSE.md` を参照してください。
