"""
把切段實驗的 CSV 結果畫成圖表（圖表文字用英文），存到 results/figures。

執行：python plot_results.py

要畫哪幾次實驗，在下方 RESULT_FILES 指定（預設是 實驗結果.md 引用的那四份）。
"""
import csv

import matplotlib

matplotlib.use("Agg")  # 不開視窗，直接存檔
import matplotlib.pyplot as plt

import experiment as exp
import rag_core as rag

# ==================== 可調整的參數 ====================
RESULTS_DIR = rag.BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# 每種文件各一份「全部設定的檢索結果」與一份「含 LLM 回答的結果」
RESULT_FILES = {
    "Structured docs": {
        "retrieval": "experiment_20261020_142058.csv",
        "answer": "experiment_20261020_143644.csv",
    },
    "Unstructured docs": {
        "retrieval": "experiment_20261020_144415_data_plain.csv",
        "answer": "experiment_20261020_150308_data_plain.csv",
    },
}
N_RETRIEVAL, N_ANSWER = 32, 36  # 檢索只算一般題；回答含陷阱題

# 顏色（已用 colorblind 檢查工具驗證過這兩色可區分）
SERIES_COLORS = {"Structured docs": "#2a78d6", "Unstructured docs": "#eb6834"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# 散佈圖上標籤要放在點左邊的（避免和旁邊的點重疊）
LABEL_LEFT = {("Structured docs", "heading-200")}
# =====================================================


def load(kind):
    """讀取每種文件的某一類結果，回傳 {文件種類: {設定名稱: 那一列}}"""
    data = {}
    for series, files in RESULT_FILES.items():
        with open(RESULTS_DIR / files[kind], encoding="utf-8-sig") as f:
            data[series] = {row["name"]: row for row in csv.DictReader(f)}
    return data


def score(text):
    """「31/32」→ 31；沒有分數（"-"）回傳 None。"""
    return int(text.split("/")[0]) if "/" in text else None


def style(ax, title, subtitle):
    """統一的圖表外觀：細格線、淡色座標軸、左上角標題。"""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(f"{title}\n", loc="left", fontsize=13, color=INK, fontweight="bold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=INK_2)


def legend(ax, **kwargs):
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kwargs)


def save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"已存 {path.relative_to(rag.BASE_DIR)}")


def dot_plot(data, column, total, title, subtitle, filename, legend_loc):
    """每個設定一列，兩種文件各一個點（上下錯開，數值相同時才不會重疊）。"""
    # 兩種文件跑過的設定取聯集，依 experiment.CONFIGS 的順序排列
    seen = {name for rows in data.values() for name in rows}
    names = [c["name"] for c in exp.CONFIGS if c["name"] in seen]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(names) + 1.4), facecolor=SURFACE)
    offsets = {"Structured docs": -0.16, "Unstructured docs": 0.16}

    values = []
    for series, rows in data.items():
        xs, ys = [], []
        for i, name in enumerate(names):
            v = score(rows[name][column]) if name in rows else None
            if v is not None:
                xs.append(v)
                ys.append(i + offsets[series])
        values += xs
        ax.scatter(xs, ys, s=64, color=SERIES_COLORS[series], edgecolors=SURFACE, linewidths=2, label=series, zorder=3)

    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    lo = min(values) - 1
    ax.set_xlim(lo, total + 0.5)
    ax.set_xticks(range(lo, total + 1))
    ax.set_xlabel(f"Correct (out of {total})", color=INK_2, fontsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax, title, subtitle)
    legend(ax, loc=legend_loc)
    save(fig, filename)


def accuracy_vs_context(answers):
    """回答正確率 vs. 每題交給 LLM 的字數：越靠左上越好（答得準又省字）。"""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    for series, rows in answers.items():
        xs = [int(r["avg_context"]) for r in rows.values()]
        ys = [score(r["answer"]) for r in rows.values()]
        ax.scatter(xs, ys, s=72, color=SERIES_COLORS[series], edgecolors=SURFACE, linewidths=2, label=series, zorder=3)
        # 同一種文件裡位置幾乎相同的點（例如 baseline 和 parent-100）合併成一個標籤
        groups = []
        for name, x, y in zip(rows, xs, ys):
            for g in groups:
                if g["y"] == y and abs(g["x"] - x) < 40:
                    g["names"].append(name)
                    break
            else:
                groups.append({"x": x, "y": y, "names": [name]})
        for g in groups:
            label = " / ".join(g["names"])
            to_left = (series, label) in LABEL_LEFT
            ax.annotate(label, (g["x"], g["y"]), xytext=(-8 if to_left else 8, -3), textcoords="offset points",
                        fontsize=8, color=INK_2, ha="right" if to_left else "left")

    ax.set_xlabel("Characters sent to the LLM per question (avg)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Answers correct (out of {N_ANSWER})", color=INK_2, fontsize=9)
    ax.set_xlim(0, None)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax, "Accuracy vs. context size", "Top-left is best: more correct answers with less context")
    legend(ax, loc="lower right")
    save(fig, "accuracy_vs_context.png")


def chunk_size_effect(retrieval, answers):
    """依標題切（heading）時，每段字數上限對檢索與回答的影響。兩張小圖，都是百分比。"""
    sizes = {"heading-100": 100, "heading-200": 200, "heading-300": 300, "baseline": 600}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    panels = [
        (axes[0], retrieval, "retrieval", N_RETRIEVAL, "Retrieval hit rate"),
        (axes[1], answers, "answer", N_ANSWER, "Answer accuracy"),
    ]
    notes = {"retrieval": "", "answer": "; 300 not run with the LLM"}
    for ax, data, column, total, title in panels:
        for series, rows in data.items():
            # 沒有跑過的大小用 NaN，線會斷開，不會假裝中間有資料
            ys = [100 * score(rows[n][column]) / total if n in rows and score(rows[n][column]) is not None else float("nan")
                  for n in sizes]
            ax.plot(list(sizes.values()), ys, color=SERIES_COLORS[series], linewidth=2, marker="o", markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=series, zorder=3)
        ax.set_xticks(list(sizes.values()))
        ax.set_xlabel("Max characters per chunk", color=INK_2, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax, title, f"heading strategy, out of {total} questions{notes[column]}")
    axes[0].set_ylabel("Correct (%)", color=INK_2, fontsize=9)
    axes[0].set_ylim(60, 102)
    legend(axes[1], loc="lower right")
    save(fig, "chunk_size_effect.png")


def main():
    retrieval, answers = load("retrieval"), load("answer")
    dot_plot(retrieval, "retrieval", N_RETRIEVAL, "Retrieval hit rate by config",
             "Retrieved chunks contain every answer keyword (32 questions, traps excluded)", "retrieval_hits.png",
             legend_loc="lower left")
    dot_plot(answers, "answer", N_ANSWER, "LLM answer accuracy by config",
             "qwen2.5:3b answer contains every keyword (36 questions incl. 4 traps)", "answer_accuracy.png",
             legend_loc="lower right")
    accuracy_vs_context(answers)
    chunk_size_effect(retrieval, answers)


if __name__ == "__main__":
    main()
