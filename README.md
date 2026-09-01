http://127.0.0.1:8080/v1

# llmcord

Discord と Local LLM を長時間無人運用するための Bot です。production 安定化の正本は https://github.com/KAFKA2306/llmcord/issues/1 です。

## Production contract

production 構成は `config.yaml` の `production` セクションを唯一の authority とします。候補 backend や汎用 provider 設定は production config に残しません。

```text
Discord
  ↓
systemd --user: llmcord.service
  ↓
llmcord
  ├─ token-aware auto compaction
  ├─ bounded queue / concurrency=1
  ├─ finite generation deadlines
  └─ deterministic watchdog
         ↓ 127.0.0.1:8080 only
systemd --user: llmcord-llama-server.service
  ↓
llama.cpp v0.3.0
commit c1d0e7a004015f23bc0233470b747b596f29b264
  ↓
ornith-ai/Ornith-1.5-9B-GGUF
Ornith-1.5-9B-Q6_K.gguf
SHA256 b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a
  ↓
NVIDIA GPU 0
```

production は **WSL2 native** です。Docker / Docker Compose は production 経路に使用しません。inference API は `127.0.0.1` にのみ bind し、Windows host や LAN へ公開しません。

FreeToken は #7 の実機比較・24h soak・無人復旧条件を満たすまで production へ昇格しません。そのため現時点の backend authority は llama.cpp です。

## 固定値

| 項目 | Production |
| --- | --- |
| backend | `llama.cpp` |
| version | `v0.3.0` |
| commit | `c1d0e7a004015f23bc0233470b747b596f29b264` |
| model | `ornith-ai/Ornith-1.5-9B` |
| GGUF repo | `ornith-ai/Ornith-1.5-9B-GGUF` |
| GGUF revision | `2b651f3` |
| artifact | `Ornith-1.5-9B-Q6_K.gguf` |
| SHA256 | `b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a` |
| quantization | `Q6_K` |
| context | `32768` tokens |
| API | `http://127.0.0.1:8080/v1` |
| supervisor | `systemd --user` |
| backend unit | `llmcord-llama-server.service` |
| bot unit | `llmcord.service` |

モデル本来の最大 context と production context は別物です。production は VRAM と無人運用の上限を明示するため `32768` に固定します。通常の長い会話は利用者に整理させず、llmcord が token-aware auto compaction します。

## セットアップ

### 1. llmcord

```bash
git clone https://github.com/KAFKA2306/llmcord.git ~/llmcord
cd ~/llmcord
uv sync --locked
uv run --locked --no-sync python production_runtime.py check --config config.yaml --offline
```

Python dependency の authority は `pyproject.toml` と `uv.lock` です。

### 2. llama.cpp

```bash
mkdir -p ~/.local/src ~/.local/opt
git clone https://github.com/ggml-org/llama.cpp.git ~/.local/src/llama.cpp-v0.3.0
cd ~/.local/src/llama.cpp-v0.3.0
git checkout c1d0e7a004015f23bc0233470b747b596f29b264
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_IS_DEV=OFF \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/llama.cpp-v0.3.0"
cmake --build build -j --target llama-server
cmake --install build
```

`latest` や未固定 `master` は production に使いません。

### 3. model artifact

```bash
mkdir -p ~/.local/share/llmcord/models
curl -fL \
  -o ~/.local/share/llmcord/models/Ornith-1.5-9B-Q6_K.gguf \
  https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF/resolve/2b651f3/Ornith-1.5-9B-Q6_K.gguf

echo 'b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a  /home/'"$USER"'/.local/share/llmcord/models/Ornith-1.5-9B-Q6_K.gguf' | sha256sum -c -
```

backend service 起動時にも `production_runtime.py` が同じ SHA256 を計算し、不一致なら `llama-server` を起動しません。

### 4. Discord secrets

```bash
mkdir -p ~/.config/llmcord
cat > ~/.config/llmcord/llmcord.env <<'EOF'
DISCORD_BOT_TOKEN=replace-me
DISCORD_CLIENT_ID=replace-me
EOF
chmod 600 ~/.config/llmcord/llmcord.env
```

Token は repository、image、service unit に埋め込みません。

### 5. systemd user services

```bash
mkdir -p ~/.config/systemd/user
ln -sfn ~/llmcord/ops/systemd/llmcord-llama-server.service ~/.config/systemd/user/llmcord-llama-server.service
ln -sfn ~/llmcord/ops/systemd/llmcord.service ~/.config/systemd/user/llmcord.service
systemctl --user daemon-reload
systemctl --user enable --now llmcord-llama-server.service llmcord.service
```

`llmcord-llama-server.service` が backend process の supervisor authority です。bot の watchdog は backend process を直接 kill / spawn せず、次だけを呼びます。

```text
systemctl --user restart llmcord-llama-server.service
```

bot 自身は `llmcord.service` が別に監督します。

## 起動時検証

static contract:

```bash
cd ~/llmcord
uv run --locked --no-sync python production_runtime.py check --config config.yaml --offline
```

実 binary と model artifact を含む検証:

```bash
uv run --locked --no-sync python production_runtime.py check --config config.yaml
```

この検証は以下を fail loudly にします。

- backend version / commit 不一致
- model artifact 不在
- model SHA256 不一致
- `0.0.0.0` など loopback 以外の bind
- `latest` / `main` など未固定 revision
- production 以外の provider / model 混在
- context window の二重 authority
- watchdog model / systemd restart interface の不一致

## Runtime control

`config.yaml` では inference concurrency を `1` に固定し、queue も有限です。

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

production context は `32768` tokens、output reservation は `4096`、safety margin は `1024` です。input hard limit は `27648` tokens です。

```text
26000 tokens: compaction trigger
18000 tokens: compaction target
4 messages: recent verbatim messages
2048 tokens: compaction summary output upper bound
```

system/developer authority、直近会話、現在入力は保護します。silent truncation は行いません。

## Health / recovery

正常性 authority は `/health` や `/v1/models` ではなく実 generation です。

```text
POST /v1/chat/completions
Reply exactly: PONG
```

連続 failure が2回に達したら受付を閉じます。systemd user service を restart し、NVIDIA GPU coarse check と synthetic generation probe の両方が成功して初めて受付を再開します。restart attempt は有限です。

厳密な backend process ↔ GPU process attribution は #10、実 Discord / 実 GPU / process kill / hang / restart / rollback / 24h soak は #14 の受入試験で確認します。CI 成功だけを production 成功とは扱いません。

## Rollback

llmcord は戻したい main commit を checkout し、unit を reload / restart します。

```bash
cd ~/llmcord
git checkout <verified-commit>
systemctl --user daemon-reload
systemctl --user restart llmcord-llama-server.service llmcord.service
```

backend/model を変更した release の rollback では、その commit の `config.yaml` に記録された llama.cpp commit と model SHA256 に戻します。`production_runtime.py check` が一致するまで service は production-ready と扱いません。

## 検証

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile \
  llmcord.py backend_probe.py context_management.py runtime_control.py health_control.py production_runtime.py
uv run --locked --no-sync python production_runtime.py check --config config.yaml --offline
uv run --locked --no-sync python -m unittest discover -s tests -v
```

## 主なファイル

```text
config.yaml                                  production / runtime authority
production_runtime.py                        contract validation / backend exec
ops/systemd/llmcord-llama-server.service     backend supervisor
ops/systemd/llmcord.service                  bot supervisor
llmcord.py                                   Discord bridge
context_management.py                        token-aware auto compaction
runtime_control.py                            queue / deadlines
backend_probe.py                              real-generation health probe
health_control.py                             watchdog / recovery
pyproject.toml                                direct dependency authority
uv.lock                                       locked dependency solution
```

## License

MIT License。`LICENSE.md` を参照してください。
