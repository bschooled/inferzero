#!/usr/bin/env bash
# Tuned 2026-11-20 on this node (Core Ultra 7 155H, 30 GB RAM).
# tg ladder measured, 40B MTP GGUFs: raw Q4 3.05 | Q4+d3 6.03 | IQ3_M+d3 7.12 | IQ2_M+d3 7.95 tok/s single user.
# IQ2_M has minor factual slips (Austen 1815 vs 1813); IQ3_M is the quality sweet spot. Q4 clean.
# --mtp-drafts 3 beats default and 5/7 (5+ drafts: acceptance falls, drafting eats 12-17% of GPU time).
# Two concurrent users: add --slots 2 (needs capped --ctx; auto OOMs slot 1). Aggregate 9.7 tok/s on IQ2.
MODEL="${1:-$HOME/models/Qwen3.8-40B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ3_M.gguf}"
cd "$HOME/inferzero"
exec ./bin/serverz -m "$MODEL" --ctx 8192 --mtp --mtp-drafts 3 --slots 2 \
  --host 0.0.0.0 --port 8080 "${@:2}"
