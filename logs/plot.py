import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import to_rgba
# ---------- 1. 读入数据 ---------

exps = [
    {
        "file_name": "loss_history_w1.0_observation_all-ep3-bsz2-lr1e-05-6-30-40-shuffle-CoF-CoM_w_MathVista-PixelReasoner-ReFocus-Zebra_CoT_count-Zebra_CoT_visual_search-Zebra_CoT_geometry_2025-08-04T07:34:36",
        "exp_name": "w/o teacher SFT"
    },
    {
        "file_name": "/home/dids/shiyang/codes/abstract-visual-token/logs/loss_history_w1.0_avt_stage1-observation_all-ep10-bsz1-lr1e-05-6-30-40_2025-08-14T22:34:11",
        "exp_name": "w teacher SFT unfreeze teacher forward"
    },
    {
        "file_name": "/home/dids/shiyang/codes/abstract-visual-token/logs/loss_history_w1.0_avt_stage1-observation_all-ep10-bsz1-lr1e-05-6-30-40_2025-08-15T12:09:07",
        "exp_name": "w teacher SFT freeze teacher forward"
    }
]

save_name = "8.15"
all_colors = ["tab:red","tab:green","tab:blue","tab:purple"]
all_lines = ['-','-.', '--']
plt.figure(figsize=(8, 5))
for i, exp in enumerate(exps):
    file_name = exp['file_name']
    df = pd.read_csv(f"{file_name}.csv")
    exp_name = exp['exp_name']
    # ---------- 2. 自动设置平滑参数 (EMA) ----------
    span = max(5, len(df) // 20)          # 至少 5，约为总长度的 5%
    df_smooth = df.ewm(span=span, adjust=False).mean()
    # ---------- 3. 绘制所有 Loss 曲线 ----------
    colors = {
        "loss_total": "tab:blue",
        "loss_student_ce": "tab:orange",
        "loss_align": "tab:red"
    }

    save_path = "./loss_imgs"
    os.makedirs(save_path, exist_ok=True)
    for col, c in colors.items():
        # 原始曲线（混合颜色）
        mixed_color = [(to_rgba("grey")[j] + to_rgba(c)[j]) / 2 for j in range(4)]
        plt.plot(df.index, df[col], color=mixed_color, alpha=0.3, linewidth=1)
        # 平滑曲线
        plt.plot(df.index, df_smooth[col], color=c, label=f"{col} {exp_name}", linestyle=all_lines[i])

plt.xlabel("global_step")
plt.ylabel("Loss value")
plt.title("align observation_end")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./loss_imgs/{save_name}_all_loss.jpg", dpi=300)

# ---------- 4. loss_align 专用对数图 ----------
plt.figure(figsize=(8, 5))

for i, exp in enumerate(exps):
    file_name = exp['file_name']
    df = pd.read_csv(f"{file_name}.csv")
    df_smooth = df.ewm(span=span, adjust=False).mean()
    exp_name = exp['exp_name']
    mixed_color = [(to_rgba("grey")[j] + to_rgba(all_colors[i])[j]) / 2 for j in range(4)]
    plt.plot(df.index, df["loss_align"], color=mixed_color, alpha=0.3, linewidth=1)
    plt.plot(df.index, df_smooth["loss_align"], color=all_colors[i], label=f"loss_align {exp_name}")
plt.yscale("log")
plt.xlabel("global_step")
plt.ylabel("Loss value (log scale)")
plt.title("observation_end loss_alignment (log scale)")
plt.legend()
plt.grid(True, which="both", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"./loss_imgs/{save_name}_loss_align_log.jpg", dpi=300)
