#!/usr/bin/env bash
# 消融实验启动脚本（修正版）
# 四组实验在同一个 window 中串行执行：
#
#   实验① Baseline       use_sae=F  use_attn=F  use_ihde=F  → ablation_base
#   实验② +Attn(无角度)  use_sae=F  use_attn=T  use_ihde=F  → ablation_attn
#   实验③ +SAE+Attn      use_sae=T  use_attn=T  use_ihde=F  → ablation_sae_attn
#   实验④ 完整模型       use_sae=T  use_attn=T  use_ihde=T  → ablation_full
#
# 注：每组均先在 OmniThings 预训练 30 epoch，再在 OmniHouse+Sunny 微调 16 epoch

set -euo pipefail

SESSION="ablation"
WINDOW="all_exps"
CONDA_ENV="romnistereo"
CONDA_SH="/usr/local/miniconda3/etc/profile.d/conda.sh"

# ── 1. 检查 tmux 是否已安装 ──────────────────────────────────────────────────
if ! command -v tmux &>/dev/null; then
    echo "[错误] tmux 未安装，请先执行: apt-get install -y tmux" >&2
    exit 1
fi

# ── 2. 检查 session 是否已存在 ───────────────────────────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[错误] tmux session '$SESSION' 已存在，请先执行: tmux kill-session -t $SESSION" >&2
    exit 1
fi

# ── 3. 确定脚本所在目录（train.py 所在目录）───────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 4. 八段命令（四组 × 预训练+微调）全部用 && 串行连接 ──────────────────────
CMD="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && cd \"$SCRIPT_DIR\" && \
\
python train.py \
  --name ablation_base \
  --dbname omnithings \
  --total_epochs 30 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --use_sae False --use_attn False --use_ihde False && \
echo \"实验①预训练完成: \$(date)\" && \
python train.py \
  --name ablation_base_ft \
  --dbname omnihouse sunny \
  --total_epochs 16 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --pretrain_ckpt ./checkpoints/ablation_base/ablation_base_e29.pth \
  --use_sae False --use_attn False --use_ihde False && \
echo \"实验①微调完成: \$(date)\" && \
\
python train.py \
  --name ablation_attn \
  --dbname omnithings \
  --total_epochs 30 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --use_sae False --use_attn True --use_ihde False && \
echo \"实验②预训练完成: \$(date)\" && \
python train.py \
  --name ablation_attn_ft \
  --dbname omnihouse sunny \
  --total_epochs 16 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --pretrain_ckpt ./checkpoints/ablation_attn/ablation_attn_e29.pth \
  --use_sae False --use_attn True --use_ihde False && \
echo \"实验②微调完成: \$(date)\" && \
\
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
echo \"实验③微调完成: \$(date)\" && \
\
python train.py \
  --name ablation_full \
  --dbname omnithings \
  --total_epochs 30 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --use_sae True --use_attn True --use_ihde True && \
echo \"实验④预训练完成: \$(date)\" && \
python train.py \
  --name ablation_full_ft \
  --dbname omnihouse sunny \
  --total_epochs 16 \
  --seed 1234 \
  --batch_size 16 \
  --accum_steps 2 \
  --pretrain_ckpt ./checkpoints/ablation_full/ablation_full_e29.pth \
  --use_sae True --use_attn True --use_ihde True && \
echo \"实验④微调完成: \$(date)\""

# ── 5. 创建 session 和 window，发送命令 ──────────────────────────────────────
echo "[信息] 创建 tmux session '$SESSION'，window '$WINDOW'..."
tmux new-session -d -s "$SESSION" -n "$WINDOW" -x 220 -y 50
tmux send-keys -t "$SESSION:$WINDOW" "$CMD" Enter

echo ""
echo "[完成] 四组消融实验已在 tmux session '$SESSION' window '$WINDOW' 中串行启动："
echo "  实验① Baseline    → ablation_base    / ablation_base_ft"
echo "  实验② +Attn       → ablation_attn    / ablation_attn_ft"
echo "  实验③ +SAE+Attn   → ablation_sae_attn / ablation_sae_attn_ft"
echo "  实验④ 完整模型    → ablation_full    / ablation_full_ft"
echo ""
echo "查看日志："
echo "  tmux attach -t $SESSION   # 进入 session"
echo "  Ctrl+B D                  # 退出但保持运行"
