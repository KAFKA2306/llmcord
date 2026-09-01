http://127.0.0.1:8080/v1

# llmcord

Discord と Local LLM を長時間無人運用する Bot。production 安定化の正本は https://github.com/KAFKA2306/llmcord/issues/1 。

## Production authority

```text
Discord
  -> systemd --user: llmcord.service
  -> production_entrypoint.py
  -> llmcord
     - token-aware auto compaction
     - bounded queue / concurrency=1
     - finite generation deadlines
     - deterministic watchdog
     - privacy-safe JSON observability
  -> http://127.0.0.1:8080/v1
  -> systemd --user: llmcord-llama-server.service
  -> production_backend.py
  -> llama.cpp v0.3.0 @ c1d0e7a004015f23bc0233470b747b596f29b264
  -> Ornith-1.5-9B-Q6_K.gguf
  -> NVIDIA GPU 0
```

production は **WSL2 native + systemd user service** の1構成のみ。Docker / Docker Compose は production 経路に置かない。inference API は WSL 内 `127.0.0.1` のみに bind する。

FreeToken は #7 の実機比較・24h soak・無人復旧条件を満たすまで昇格しない。現時点の backend authority は llama.cpp。

| 項目 | 固定値 |
| --- | --- |
| backend | `llama.cpp` |
| release | `v0.3.0` |
| commit | `c1d0e7a004015f23bc0233470b747b596f29b264` |
| model | `ornith-ai/Ornith-1.5-9B` |
| GGUF repo | `ornith-ai/Ornith-1.5-9B-GGUF` |
| revision | `2b651f3` |
| artifact | `Ornith-1.5-9B-Q6_K.gguf` |
| SHA256 | `b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a` |
| context | `32768` tokens |
| endpoint | `http://127.0.0.1:8080/v1` |
| supervisor | `systemd --user` |
| image input | disabled (`max_images: 0`) |

`min_vram_used_mib: 1` は実測 floor の代用品ではない。未計測値を推測で authority にせず、selected GPU 上に expected `llama-server` compute process が存在することを residency の主要条件にする。実測 VRAM floor は #10 で更新する。

## Install

```bash
git clone https://github.com/KAFKA2306/llmcord.git ~/llmcord
cd ~/llmcord
uv sync --locked
uv run --locked --no-sync python production_backend.py check --static
```

### llama.cpp

```bash
mkdir -p ~/.local/src ~/.local/opt
git clone https://github.com/ggml-org/llama.cpp.git ~/.local/src/llama.cpp-v0.3.0
cd ~/.local/src/llama.cpp-v0.3.0
git checkout c1d0e7a004015f23bc0233470b747b596f29b264
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_IS_DEV=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/opt/llama.cpp-v0.3.0"
cmake --build build -j --target llama-server
cmake --install build
```

`production_backend.py check` は `llama-server --version` の exact `0.3.0` と build commit を照合する。`latest` / `main` / `master` は使わない。

### Model

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

backend service は起動前に同じ SHA256 を再計算する。不一致なら `llama-server` を起動しない。

### Discord secret

```bash
mkdir -p ~/.config/llmcord
printf '%s\n' 'DISCORD_BOT_TOKEN=replace-me' > ~/.config/llmcord/llmcord.env
chmod 600 ~/.config/llmcord/llmcord.env
```

必要な場合だけ `DISCORD_CLIENT_ID` を追加する。secret は repository / service unit に埋め込まない。

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

backend supervisor authority は `llmcord-llama-server.service`。watchdog は process を直接 spawn/kill せず `systemctl --user restart llmcord-llama-server.service` のみを呼ぶ。bot は `llmcord.service` が別に監督する。

## Validation

Static manifest:

```bash
uv run --locked --no-sync python production_backend.py check --static
```

実 binary + GGUF:

```bash
uv run --locked --no-sync python production_backend.py check
```

fail loudly にする対象:

- backend release / commit / executable drift
- model repo / revision / URL / filename / SHA256 drift
- model file missing
- wildcard / non-loopback endpoint
- production provider / model の複数化
- runtime context drift
- watchdog model / restart / GPU projection drift
- `mmproj` 未固定での image input 有効化

### WSL2 runtime evidence

#13 の実機 startup/restart 証拠は `runtime_evidence.py` で取得する。CI や mock を実機 evidence として代用しない。

```bash
cd ~/llmcord
git switch main
git pull --ff-only
uv sync --locked

uv run --locked --no-sync python runtime_evidence.py \
  --exercise-restart \
  --output evidence/runtime.json
```

collector は次を自動確認する。

- git が `main` かつ clean
- WSL2 であること
- pinned backend binary / release / commit
- pinned GGUF SHA256
- systemd user backend / bot service が `active/running`
- selected GPU 上に expected `llama-server` compute process が存在
- synthetic `/v1/chat/completions` が exact `PONG`
- manifest 固定の backend restart command 後に backend PID が変わる
- backend-only restart 中に bot PID が変わらない
- restart 後も GPU attribution + synthetic generation が再び healthy

結果は machine-readable JSON。`pass_with_rollback_unverified` は startup/restart が実証済みで、rollback だけが未実証という意味。collector 自体は git checkout を自動実行しない。

既知の旧 verified runtime を rollback target として記録する場合:

```bash
uv run --locked --no-sync python runtime_evidence.py \
  --exercise-restart \
  --rollback-commit <verified-main-commit> \
  --output evidence/runtime-before-rollback.json
```

この指定だけでは rollback 済みとは判定しない。下記 rollback 手順を実行後、旧 commit 側でも collector を再実行し、その JSON を rollback evidence とする。

## Runtime contract

```text
max_concurrency = 1
max_queue_size = 4
queue_wait_timeout = 300 s
connect_timeout = 10 s
first_generation_signal_timeout = 300 s
stream_idle_timeout = 120 s
total_generation_timeout = 1800 s

context_window = 32768
max_output = 4096
safety_margin = 1024
hard_input_limit = 27648
compaction_trigger = 26000
compaction_target = 18000
recent_verbatim_messages = 4
```

正常性 authority は `/health` / `/v1/models` ではなく実 `/v1/chat/completions` generation (`Reply exactly: PONG`)。restart 後は selected GPU 上の expected backend process と synthetic generation probe が成功して初めて受付を再開する。

## Observability

`production_entrypoint.py` は Bot より先に structured logging を設定する。主要eventは `production.startup`, `request.*`, `queue.*`, `context.compacted`, `generation.*`, `probe.success`, `watchdog.*`。

Discord message本文、prompt、system prompt、attachment本文、response本文、API key、Authorization、Discord token、user ID は観測ログに出さない。分類不能 log も元本文を捨てる。この event stream は #14 の障害注入・soak evidence に使う。

#14 の failure-injection / 24h soak evidence は `acceptance.py` が機械的に採点する。`runtime_evidence.py` は #13 の実 runtime identity/startup/restart の収集器で、責務を重複させない。

## Rollback

```bash
cd ~/llmcord
git checkout <verified-main-commit>
uv sync --locked
uv run --locked --no-sync python production_backend.py check
systemctl --user daemon-reload
systemctl --user restart llmcord-llama-server.service llmcord.service

uv run --locked --no-sync python runtime_evidence.py \
  --exercise-restart \
  --output evidence/runtime-after-rollback.json
```

その commit の `config.yaml` に固定された backend commit / model SHA256 と実体が一致し、rollback 後の runtime evidence が startup/restart `pass` になるまで production-ready と扱わない。

## CI

```bash
uv lock --check
uv sync --locked
uv run --locked --no-sync python -m py_compile \
  llmcord.py backend_probe.py context_management.py runtime_control.py health_control.py observability.py \
  production_contract.py production_entrypoint.py production_backend.py acceptance.py runtime_evidence.py
uv run --locked --no-sync python production_backend.py check --static
uv run --locked --no-sync python -m unittest discover -s tests -v
```

CIだけでは #13 を完全に閉じない。実 WSL2/GPU で startup / restart / rollback を確認し、最終 Discord E2E / failure injection / 24h soak は #14 で行う。
