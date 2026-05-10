#!/usr/bin/env bash
# 实验③：+SAE+Attn
# 预训练（OmniThings 30 epoch）→ 微调（OmniHouse+Sunny 16 epoch）串行执行

set -euo pipefail

SESSION="exp3"
WINDOW="sae_attn"
CONDA_ENV="romnistereo"
CONDA_SH="/usr/local/miniconda3/etc/profile.d/conda.sh"

# ── 检查 tmux ────────────────────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    echo "[错误] tmux 未安装" >&2; exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[错误] tmux session '$SESSION' 已存在，请先执行: tmux kill-session -t $SESSION" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$SCRIPT_DIR\" && \
python train.py \
  --name ablation_sae_attn \
  --dbname omnithings \
  --total_epochs 30 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --use_sae True --use_attn True --use_ihde False && \
echo \"实验③预训练完成: \$(date)\" && \
python train.py \
  --name ablation_sae_attn_ft \
  --dbname omnihouse sunny \
  --total_epochs 16 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --pretrain_ckpt ./checkpoints/ablation_sae_attn/ablation_sae_attn_e29.pth \
  --use_sae True --use_attn True --use_ihde False && \
echo \"实验③微调完成: \$(date)\""

echo "[信息] 创建 tmux session '$SESSION'，window '$WINDOW'..."
tmux new-session -d -s "$SESSION" -n "$WINDOW" -x 220 -y 50
tmux send-keys -t "$SESSION:$WINDOW" "$CMD" Enter

echo ""
echo "[完成] 实验③已启动："
echo "  预训练 → 微调，全部串行"
echo ""
echo "  tmux attach -t $SESSION   # 查看进度"
echo "  Ctrl+B D                  # 退出但保持运行"
