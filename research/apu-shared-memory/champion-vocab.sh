#!/usr/bin/env bash
# The iteration-1 champion line (measured 2026-11-22, LADDER-P4.md): IQ2_M +
# MTP d3 + the reduced-vocabulary draft head: 8.19 tok/s paired (+8.9 % over
# flags-off in the same session), acceptance -3.7 pp. Needs the ranked
# vocab list (scripts/draft-vocab/), 115 MB at 32,768 rows.
MODEL="${1:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf}"
exec "$(dirname "$0")/champion.sh" "$MODEL" --mtp-draft-vocab 32768 "${@:2}"
