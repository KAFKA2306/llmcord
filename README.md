# llmcord

Discord を OpenAI `/v1/chat/completions` 互換 LLM のフロントエンドとして使う Bot です。ローカル LLM とリモート LLM の両方を扱えます。

## このリポジトリについて

このリポジトリは `KAFKA2306/llmcord` として運用しています。元の実装は [jakobdylanc/llmcord](https://github.com/jakobdylanc/llmcord) です。ライセンスは MIT です。

長時間の無人運用に必要な timeout、bounded queue、実生成 probe、watchdog、GPU 状態確認などの安定化は [Issue #1](https://github.com/KAFKA2306/llmcord/issues/1) で管理しています。現行コードにはこれらの仕組みがまだすべて実装されているわけではありません。

## 主な機能

- Discord のメンション、返信、DM から LLM と会話
- 返信チェーンを会話履歴として利用し、会話用データベースは不要
- スレッド内の会話に対応
- `/model` で管理者がモデルを切り替え
- OpenAI `/v1/chat/completions` 互換 API に対応
- OpenRouter、OpenAI、xAI、Google、Azure OpenAI の設定例を同梱
- Ollama、LM Studio、vLLM のローカル API 設定例を同梱
- テキストファイル添付に対応
- 対応モデルでは画像添付に対応
- 応答をストリーミング表示し、長文は複数メッセージへ分割
- `config.yaml` をモデル選択時やメッセージ処理時に再読込
- ユーザー、ロール、チャンネル単位のアクセス制御
- `asyncio` と非同期 HTTP クライアントを使用

## 構成

```text
Discord
  |
  v
llmcord.py
  |
  | OpenAI /v1/chat/completions
  v
LLM API
  |- Ollama
  |- LM Studio
  |- vLLM
  |- その他の OpenAI 互換 API
  `- リモート LLM API
```

会話履歴は Discord の返信関係から再構築します。Bot 側に会話データベースはありません。

## 必要なもの

- Python が実行できる環境、または Docker
- Discord Bot Token
- Discord Application の Client ID
- Discord Developer Portal で有効化した `MESSAGE CONTENT INTENT`
- 使用する LLM の OpenAI 互換 API endpoint

## セットアップ

### 1. Clone

```bash
git clone https://github.com/KAFKA2306/llmcord.git
cd llmcord
```

### 2. Discord Bot を設定

[Discord Developer Portal](https://discord.com/developers/applications) で Application と Bot を作成します。

Bot の設定で `MESSAGE CONTENT INTENT` を有効にし、Token と Client ID を環境変数へ設定します。

```bash
export DISCORD_BOT_TOKEN='...'
export DISCORD_CLIENT_ID='...'
```

`.env` を使う場合も同じ変数名を使用できます。

```dotenv
DISCORD_BOT_TOKEN=...
DISCORD_CLIENT_ID=...
```

Token を `config.yaml` や Git 管理対象ファイルへ直接書かないでください。

### 3. LLM を設定

`config.yaml` の `providers` と `models` を使用します。

設定名の末尾を `_env` にすると、その値を名前とする環境変数から読み込みます。

```yaml
bot_token_env: DISCORD_BOT_TOKEN
client_id_env: DISCORD_CLIENT_ID
```

ローカル LLM の既定設定例は次のとおりです。

```yaml
providers:
  lmstudio:
    base_url: http://localhost:1234/v1

  ollama:
    base_url: http://localhost:11434/v1

  vllm:
    base_url: http://localhost:8000/v1
```

モデルは `<provider>/<model>` 形式で登録します。

```yaml
models:
  ollama/llama4:
```

`providers` に任意の OpenAI `/v1/chat/completions` 互換 API を追加することもできます。

```yaml
providers:
  local:
    base_url: http://127.0.0.1:8080/v1

models:
  local/<APIが受け付けるモデル名>:
```

`models` の先頭が起動時の既定モデルです。

### 4. 起動

Python で直接起動する場合:

```bash
python -m pip install -U -r requirements.txt
python llmcord.py
```

Docker Compose を使う場合:

```bash
docker compose up --build
```

現行 `docker-compose.yaml` は `network_mode: host` と `restart: unless-stopped` を使用します。ローカル LLM へ接続できない場合は、llmcord を実行している環境から `base_url` へ実際に到達できるか確認してください。

## Discord 側の使い方

### サーバー

Bot をメンションすると新しい会話を開始します。

```text
@bot 質問
```

Bot の返答へ返信すると、その返信チェーンを会話履歴として続行します。任意のメッセージへ返信しながら Bot をメンションすると、そのメッセージから会話を開始できます。

### DM

`allow_dms: true` の場合、DM では毎回メンションしなくても会話が継続します。新しい会話を開始したい場合は Bot をメンションします。

### スレッド

既存メッセージから Discord スレッドを作成し、スレッド内で Bot をメンションすると会話を続けられます。

### モデル切り替え

```text
/model
```

`permissions.users.admin_ids` に登録したユーザーだけがモデルを変更できます。

## 設定

### Discord

| 設定 | 内容 | 既定値 |
| --- | --- | --- |
| `bot_token` | Discord Bot Token | 必須 |
| `client_id` | Discord Application Client ID | 未設定 |
| `status_message` | Bot のステータスメッセージ。最大128文字 | 未設定 |
| `max_text` | 1メッセージから LLM へ渡す最大文字数。テキスト添付を含む | `100000` |
| `max_images` | 1メッセージから渡す最大画像数 | `5` |
| `max_messages` | 返信チェーンから使用する最大メッセージ数 | `25` |
| `use_plain_responses` | Embed ではなく plaintext component を使う | `false` |
| `allow_dms` | DM を許可する | `true` |
| `permissions` | user / role / channel ごとの許可・拒否 | 全許可相当 |

`allowed_ids` が空の場合、その分類では allowlist 制限を行いません。`blocked_ids` は拒否対象です。`admin_ids` のユーザーは `/model` を使用できます。

### LLM

| 設定 | 内容 |
| --- | --- |
| `providers` | provider 名ごとの `base_url`、`api_key`、追加 HTTP 設定 |
| `models` | `<provider>/<model>` ごとのモデル設定 |
| `system_prompt` | 全会話へ追加する system prompt |

provider では必要に応じて次の項目を使用できます。

- `api_key` / `api_key_env`
- `extra_headers`
- `extra_query`
- `extra_body`

モデル側へ設定した値は request の追加 body として送られます。

`system_prompt` では `{date}` と `{time}` を使用でき、ホストのローカル時刻で置換されます。

## 画像と添付ファイル

テキストまたは画像として認識された添付ファイルだけを処理します。

画像入力はモデル名から vision 対応を推定します。必要な場合はモデル名の末尾へ `:vision` を付けることで画像入力を有効にできます。

上限を超えた入力や未対応添付がある場合は、通常の Embed 応答では警告を表示します。

## 現行実装の運用上の注意

現時点のコードは通常の対話用途を中心とした構成です。Local LLM を長期間無人運用する場合、次の機能はまだ production contract として確立されていません。

- LLM request の明示的な有限 timeout
- GPU inference の bounded queue
- 同時生成数の強制上限
- generation path を通す synthetic health probe
- backend freeze 時の deterministic watchdog
- restart storm 防止
- GPU 消失や CPU fallback の異常判定
- request_id を使った end-to-end の追跡

これらは [Issue #1](https://github.com/KAFKA2306/llmcord/issues/1) で実装・検証します。

また、既定の `max_text: 100000` と `max_messages: 25` はすべての Local LLM に安全な値ではありません。実際の context length と VRAM に合わせて制限してください。

## トラブルシュート

Bot がメッセージを読まない場合:

1. Discord Developer Portal で `MESSAGE CONTENT INTENT` が有効か確認する
2. サーバーでは Bot をメンションしているか確認する
3. `permissions` の allowlist / blocklist を確認する

LLM に接続できない場合:

1. `providers.<name>.base_url` を確認する
2. llmcord の実行環境から endpoint へ到達できるか確認する
3. `<provider>/<model>` の provider 名が `providers` に存在するか確認する
4. API key が必要な provider では環境変数が設定されているか確認する

生成中の例外は現在ログへ記録されます。無人運用での自動復旧は Issue #1 の対象です。

## ファイル

```text
llmcord.py           Discord Bot 本体
config.yaml          実行設定と provider / model 定義
requirements.txt     Python 依存関係
Dockerfile           コンテナイメージ
Dockerfile           
docker-compose.yaml  Compose 実行設定
LICENSE.md            MIT License
README.md             この文書
```

## Upstream / License

元のプロジェクト:

https://github.com/jakobdylanc/llmcord

MIT License。著作権表示とライセンス本文は `LICENSE.md` を参照してください。
