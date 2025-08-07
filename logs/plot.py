import pandas as pd
import matplotlib.pyplot as plt
import os
# ---------- 1. 读入数据 ---------
file_name = "loss_history_w1.0_observation_all-ep3-bsz2-lr1e-05-6-30-40-shuffle-CoF-CoM_w_MathVista-PixelReasoner-ReFocus-Zebra_CoT_count-Zebra_CoT_visual_search-Zebra_CoT_geometry_2025-08-04T07:34:36"

df = pd.read_csv(f"{file_name}.csv")

# ---------- 2. 自动设置平滑参数 (EMA) ----------
span = max(5, len(df) // 20)          # 至少 5，约为总长度的 5%
df_smooth = df.ewm(span=span, adjust=False).mean()

# ---------- 3. 绘制所有 Loss 曲线 ----------
plt.figure(figsize=(8, 5))
colors = {
    "loss_total": "tab:blue",
    "loss_student_ce": "tab:orange",
    "loss_teacher_ce": "tab:green",
    "loss_align": "tab:red"
}

save_path = "./loss_imgs"
os.makedirs(save_path, exist_ok=True)
for col, c in colors.items():
    # 原始曲线（淡化）
    plt.plot(df.index, df[col], color=c, alpha=0.3, linewidth=1)
    # 平滑曲线
    plt.plot(df.index, df_smooth[col], color=c, label=f"{col} (smoothed)")

plt.xlabel("global_step")
plt.ylabel("Loss value")
plt.title("align observation_end")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./loss_imgs/{file_name}.jpg", dpi=300)

# ---------- 4. loss_align 专用对数图 ----------
plt.figure(figsize=(8, 5))
plt.plot(df.index, df["loss_align"], color="grey", alpha=0.3, linewidth=1, label="loss_align (raw)")
plt.plot(df.index, df_smooth["loss_align"], color="tab:red", label="loss_align (smoothed)")
plt.yscale("log")
plt.xlabel("global_step")
plt.ylabel("Loss value (log scale)")
plt.title("observation_end loss_alignment (log scale)")
plt.legend()
plt.grid(True, which="both", linewidth=0.5)
plt.tight_layout()
plt.savefig(f"./loss_imgs/{file_name}_loss_align_log.jpg", dpi=300)
