http://127.0.0.1:8080/v1

# llmcord

Discord と Local LLM を長時間無人運用するための Bot です。production 安定化の正本は https://github.com/KAFKA2306/llmcord/issues/1 です。

## Production

現在の production は次の1構成だけです。

```text
Discord
  ↓
systemd --user: llmcord.service
  ↓
production_entrypoint.py
  ↓
llmcord
  ├─ token-aware auto compaction
  ├─ bounded queue / concurrency=1
  ├─ finite generation deadlines
  ├─ deterministic watchdog
  └─ privacy-safe JSON observability
         ↓ http://127.0.0.1:8080/v1
systemd --user: llmcord-llama-server.service
  ↓
production_backend.py
  ↓
llama.cpp v0.3.0
commit c1d0e7a004015f23bc0233470b747b596f29b264
  ↓
Ornith-1.5-9B-Q6_K.gguf
SHA256 b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a
  ↓
NVIDIA GPU 0
```

production は **WSL2 native + systemd user service** です。Docker / Docker Compose は production 経路から削除しています。inference API は WSL 内 `127.0.0.1` のみに bind し、Windows host や LAN へ公開しません。

FreeToken は #7 の実機比較・24h soak・無人復旧条件を満たすまで production へ昇格しません。現時点の backend authority は llama.cpp です。

### 固定値

| 項目 | 値 |
| --- | --- |
| backend | `llama.cpp` |
| release | `v0.3.0` |
| commit | `c1d0e7a004015f23bc0233470b747b596f29b264` |
| model upstream | `ornith-ai/Ornith-1.5-9B` |
| artifact repo | `ornith-ai/Ornith-1.5-9B-GGUF` |
| revision | `2b651f3` |
| artifact | `Ornith-1.5-9B-Q6_K.gguf` |
| SHA256 | `b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a` |
| context | `32768` tokens |
| endpoint | `http://127.0.0.1:8080/v1` |
| supervisor | `systemd --user` |
| backend service | `llmcord-llama-server.service` |
| bot service | `llmcord.service` |
| image input | disabled (`max_images: 0`) |

モデル自体の最大 context と production context は別です。production は VRAM と長時間運用の上限を明示するため `32768` に固定し、長い会話は llmcord が token-aware auto compaction します。

画像入力は `mmproj` の revision / SHA256 / VRAM 条件を production contract に固定するまで有効化しません。

`min_vram_used_mib` は現時点で実GPU上の定常値を未計測のため `1` にしています。これは「GPU上に何らかの allocation がある」ための最低 guard であり、model residency の判定 authority は selected GPU 上の expected `llama-server` compute process です。実測 floor は #10 の実GPU evidence で更新します。

## セットアップ

### llmcord

```bash
git clone https://github.com/KAFKA2306/llmcord.git ~/llmcord
cd ~/llmcord
uv sync --locked
uv run --locked --no-sync python production_backend.py check --static
```

依存関係の正本は `pyproject.toml` と `uv.lock` です。

### llama.cpp

```bash
mkdir -p ~/.local/src ~/.local/opt
git clone https://github.com/ggml-org/llama.cpp.git ~/.local/src/llama.cpp-v0.3.0
cd ~/.local/src/llama.cpp-v0.3.0
git checkout c1d0e7a004015f23bc0233470b747b596f29b264
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/llama.cpp-v0.3.0"
cmake --build build -j --target llama-server
cmake --install build
```

`latest` / `main` / `master` は production に使いません。

### model artifact

```bash
mkdir -p ~/.local/share/llmcord/models
curl -fL \
  -o ~/.local/share/llmcord/models/Ornith-1.5-9B-Q6_K.gguf \
  https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/2b651f3/Ornith-1.5-9B-Q6_K.gguf

printf '%s  %s\n' \
  'b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a' \
  "$HOME/.local/share/llmcord/models/Ornith-1.5-9B-Q6_K.gguf" \
  | sha256sum -c -
```

`production_backend.py check` と backend service 起動時にも同じ artifact hash を検証します。不一致なら backend を起動しません。

### Discord token

```bash
mkdir -p ~/.config/llmcord
cat > ~/.config/llmcord/llmcord.env <<'EOF'
DISCORD_BOT_TOKEN=replace-me
EOF
chmod 600 ~/.config/llmcord/llmcord.env
```

必要な場合のみ `DISCORD_CLIENT_ID` を追加します。secret は repository や service unit に埋め込みません。

### systemd user services

```bash
systemctl --user is-system-running
sudo loginctl enable-linger "$USER"

mkdir -p ~/.config/systemd/user
ln -sfn ~/llmcord/ops/systemd/llmcord-llama-server.service ~/.config/systemd/user/llmcord-llama-server.service
ln -sfn ~/llmcord/ops/systemd/llmcord.service ~/.config/systemd/user/llmcord.service
systemctl --user daemon-reload
systemctl --user enable --now llmcord-llama-server.service llmcord.service
```

`systemctl --user` が利用できない環境はこの production contract と一致しません。別 supervisor へ silent fallback しません。

backend process の supervisor authority は `llmcord-llama-server.service` です。watchdog は process を直接 spawn / kill せず、次だけを実行します。

```text
systemctl --user restart llmcord-llama-server.service
```

bot 自身は `llmcord.service` が別に監督します。

## 検証

manifest だけを検証:

```bash
cd ~/llmcord
uv run --locked --no-sync python production_backend.py check --static
```

実 binary の release/commit と実 GGUF SHA256 まで検証:

```bash
uv run --locked --no-sync python production_backend.py check
```

この検証は以下を fail loudly にします。

- backend release / commit / executable の不一致
- model repo / revision / artifact URL / filename / SHA256 の不一致
- model file 不在
- endpoint の wildcard / non-loopback 化
- production provider / model の複数化
- runtime context window の drift
- watchdog model / restart command / GPU 条件の drift
- `mmproj` 未固定状態での画像入力有効化

## Runtime control

```text
max_concurrency = 1
max_queue_size = 4
queue_wait_timeout = 300 s
connect_timeout = 10 s
first_generation_signal_timeout = 300 s
stream_idle_timeout = 120 s
total_generation_timeout = 1800 s
```

watchdog が `degraded` / `recovering` に入った場合、新規 request と queue 待ち request の両方を backend へ送りません。

## Context management

```text
context_window = 32768
max_output = 4096
safety_margin = 1024
hard input limit = 27648
compaction trigger = 26000
compaction target = 18000
recent verbatim messages = 4
compaction summary max output = 2048
```

system/developer authority、直近会話、現在入力は保持し、silent truncation は行いません。

## Health / recovery

正常性 authority は `/health` や `/v1/models` ではなく実 generation です。

```text
POST /v1/chat/completions
Reply exactly: PONG
```

連続 failure が2回に達したら受付を閉じ、systemd backend service を restart します。restart 後は NVIDIA GPU の selected device / expected `llama-server` compute process と synthetic generation probe が成功して初めて受付を再開します。

厳密な実GPU検証は #10、実 Discord / process kill / generation hang / rollback / 24h soak は #14 で確認します。CI 成功だけを production 成功とは扱いません。

## Production observability

`production_entrypoint.py` は Bot 本体より先に structured logging を設定し、運用ログを1行1JSON objectへ変換します。production event には固定した provider / model と request ID を付与でき、#14 の failure injection / soak evidence に使用します。

主要event:

```text
production.startup
request.received
request.rejected
queue.admitted
queue.rejected
queue.timeout
context.compacted
generation.timeout
generation.failure
probe.success
watchdog.failure
watchdog.restart
watchdog.recovered
log
```

Discord message本文、prompt、system prompt、attachment本文、response本文、API key、Authorization、Discord token、user ID は観測ログへ出しません。分類不能な runtime log も元の message 本文を捨てて fail-closed redaction します。

## Rollback

```bash
cd ~/llmcord
git checkout <verified-main-commit>
uv sync --locked
uv run --locked --no-sync python production_backend.py check
systemctl --user daemon-reload
systemctl --user restart llmcord-llama-server.service llmcord.service
```

backend/modelを変更したreleaseのrollbackでは、そのcommitの `config.yaml` に記録された llama.cpp commit と model SHA256 に実体を戻します。`production_backend.py check` が成功するまで production-ready と扱いません。

## CI

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile \
  llmcord.py backend_probe.py context_management.py runtime_control.py health_control.py observability.py \
  production_contract.py production_entrypoint.py production_backend.py
uv run --locked --no-sync python production_backend.py check --static
uv run --locked --no-sync python -m unittest discover -s tests -v
```

## 主なファイル

```text
config.yaml                                  canonical production/runtime manifest
production_contract.py                       contract validation
production_backend.py                        backend check / startup
production_entrypoint.py                      bot fail-closed startup
ops/systemd/llmcord-llama-server.service     backend supervisor
ops/systemd/llmcord.service                  bot supervisor
health_control.py                             watchdog / GPU process health
runtime_control.py                            bounded queue / deadlines
context_management.py                        token-aware auto compaction
backend_probe.py                              real-generation health probe
observability.py                              privacy-safe production JSON events
```

## License

MIT License。`LICENSE.md` を参照してください。
