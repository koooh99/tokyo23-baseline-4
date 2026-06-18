# 06_report.py
# 結果を「単年の一発R²」ではなく「時点をまたいだ分布」で正直に可視化する。
# 年次(baseline_summary.csv)と四半期(baseline_summary_q.csv)の両方を図にし、
# さらに両者を直接並べた比較図を出す（既定ラグ構成は四半期 Qboth）。
import matplotlib.pyplot as plt
import pandas as pd
import config as C

try:
    import japanize_matplotlib  # noqa
except ImportError:
    pass

print("=== 6. レポート（分布で示す：年次 / 四半期 / 比較） ===")


def overall_fig(summary, title, out_name):
    """lag × model の全体R²を平均±標準偏差の棒グラフにする。"""
    allrows = summary[summary["scope"] == "ALL"].copy()
    if allrows.empty:
        return
    allrows["key"] = allrows["lag"] + "/" + allrows["model"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(allrows["key"], allrows["r2_mean"], yerr=allrows["r2_std"],
           capsize=5, color="steelblue", alpha=0.85)
    ax.set_ylabel("R² (mean ± std across folds)")
    ax.set_title(title)
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def subgroup_fig(summary, lag_name, title, out_name):
    """都心5 / その他 のXGB R²比較（指定ラグ構成）。"""
    grp = summary[(summary["scope"] == "GROUP") & (summary["model"] == "XGB")
                  & (summary["lag"] == lag_name)]
    if grp.empty:
        print(f"  [skip] subgroup図: lag={lag_name} が無い ({out_name})")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grp["name"], grp["r2_mean"], yerr=grp["r2_std"], capsize=5,
           color=["darkorange", "seagreen"], alpha=0.85)
    ax.set_ylabel("R² (mean ± std)")
    ax.set_title(title)
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)


# 1) 年次（既存の図を維持：消さない） --------------------------------------
yearly_path = C.OUT_DIR / "baseline_summary.csv"
if yearly_path.exists():
    sy = pd.read_csv(yearly_path)
    overall_fig(sy, "Yearly lag baseline: R² is a distribution, not a single year",
                "fig_overall_r2.png")
    subgroup_fig(sy, "Lag1", "Central-5 vs Others (XGB, yearly Lag1)",
                 "fig_subgroups.png")
    print("  年次: fig_overall_r2.png, fig_subgroups.png")

# 2) 四半期（Qboth を既定として強調） --------------------------------------
q_path = C.OUT_DIR / "baseline_summary_q.csv"
sq = None
if q_path.exists():
    sq = pd.read_csv(q_path)
    overall_fig(sq, "Quarterly lag baseline (Qprev/Qyoy/Qboth): R² distribution",
                "fig_overall_r2_q.png")
    subgroup_fig(sq, C.DEFAULT_LAG_NAME,
                 f"Central-5 vs Others (XGB, quarterly {C.DEFAULT_LAG_NAME})",
                 "fig_subgroups_q.png")
    print("  四半期: fig_overall_r2_q.png, fig_subgroups_q.png")

# 3) 年次 vs 四半期 の直接比較（XGB全体R²） --------------------------------
if yearly_path.exists() and sq is not None:
    ay = sy[(sy["scope"] == "ALL") & (sy["model"] == "XGB")].copy()
    aq = sq[(sq["scope"] == "ALL") & (sq["model"] == "XGB")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x_y = range(len(ay))
    x_q = range(len(ay), len(ay) + len(aq))
    ax.bar(list(x_y), ay["r2_mean"], yerr=ay["r2_std"], capsize=5,
           color="slategray", alpha=0.85, label="年次 (yearly)")
    ax.bar(list(x_q), aq["r2_mean"], yerr=aq["r2_std"], capsize=5,
           color="steelblue", alpha=0.85, label="四半期 (quarterly)")
    # 既定構成(Qboth)の棒を強調
    for i, name in zip(x_q, aq["lag"]):
        if name == C.DEFAULT_LAG_NAME:
            ax.patches[i].set_color("crimson")
            ax.patches[i].set_alpha(0.9)
    labels = list(ay["lag"]) + list(aq["lag"])
    ax.set_xticks(list(x_y) + list(x_q))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("R² (mean ± std across folds)")
    ax.set_title("年次 vs 四半期ラグ：全体R²の比較 (XGB)  ※赤=既定 Qboth")
    ax.axhline(0, color="black", lw=0.6)
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(C.OUT_DIR / "fig_year_vs_quarter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  比較: fig_year_vs_quarter.png")

    # 数値の比較表も併せて吐く（論文の表用）
    ay["granularity"] = "yearly"
    aq["granularity"] = "quarterly"
    comp = pd.concat([ay, aq], ignore_index=True)[
        ["granularity", "lag", "model", "r2_mean", "r2_std",
         "r2_trim_mean", "r2_trim_std", "n_folds"]].round(3)
    comp.to_csv(C.OUT_DIR / "year_vs_quarter_overall.csv",
                index=False, encoding="utf-8-sig")
    print("  比較表: year_vs_quarter_overall.csv")
    print(comp.to_string(index=False))

print("完了")
