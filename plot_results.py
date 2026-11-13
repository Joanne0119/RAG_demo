"""
把切段實驗的 CSV 結果畫成圖表（圖表文字用英文），存到 results/figures。

執行：python plot_results.py

要畫哪幾次實驗，在下方 RESULT_FILES（自己寫的文件）和 REAL_FILES（真實說明書 PDF）指定。
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

# 真實說明書 PDF（結果四）：檢索跑全部設定，回答只跑 6 組
REAL_FILES = {
    "Car manual (zh)": {
        "retrieval": "experiment_20261109_141413_data_car.csv",
        "answer": "experiment_20261112_210016_data_car.csv",
    },
    "RP2040 datasheet (en)": {
        "retrieval": "experiment_20261109_135416_data_rp2040.csv",
        "answer": "experiment_20261113_001918_data_rp2040.csv",
    },
}

# 顏色（已用 colorblind 檢查工具驗證過可區分）。真實 PDF 用第 3、4 色，不和前兩種文件混淆
SERIES_COLORS = {
    "Structured docs": "#2a78d6", "Unstructured docs": "#eb6834",
    "Car manual (zh)": "#1baf7a", "RP2040 datasheet (en)": "#eda100",
}
# 第 3、4 色比較淡，再用不同形狀區分（色盲或黑白列印時也看得出來）
SERIES_MARKERS = {"RP2040 datasheet (en)": "s"}  # 沒列的用圓點
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# 散佈圖上標籤要放在點左邊的（避免和旁邊的點重疊）
LABEL_LEFT = {("Structured docs", "heading-200")}
REAL_LABEL_LEFT = set()

# 逐題熱力圖用的「含答錯題號」的 LLM 結果（experiment.py 有 wrong 欄位之後才有）
DETAIL_FILES = {
    "Structured docs": "experiment_20261022_021218.csv",
    "Unstructured docs": "experiment_20261022_022917_data_plain.csv",
}

# 每題的英文短標籤，順序和 evaluate.py 的 TESTS 一樣
QUESTION_LABELS = [
    "E03 error code", "Descaling interval", "Descaling duration", "App can't find machine",
    "Coffee too weak", "Warranty length", "Ground coffee amount", "Milk froth temperature",
    "Registration deadline", "Team size", "Outside student as leader", "Registration fee",
    "1st prize & tax", "Code submission deadline", "Tech completeness weight",
    "Kids using the machine", "Placing against a wall", "Dual-band router", "Adjust grind anytime",
    "Graduates eligible", "Latte default volume", "Double espresso max", "2nd prize amount",
    "Filter replacement", "Support closing time", "Contact phone extension", "E05 & E09 fixes",
    "Final venue & start time", "Fee and deposit", "Longest warranty", "Warranty exclusions",
    "Number of finalist teams",
    "Iced coffee", "Parking", "Machine price", "Travel subsidy",
]
# 題型分組：(名稱, 第一題, 最後一題)
QUESTION_GROUPS = [
    ("Basic", 1, 15), ("Paraphrase", 16, 20), ("Table detail", 21, 24),
    ("Confusable", 25, 27), ("Multi-part / reasoning", 28, 32), ("Trap (no answer)", 33, 36),
]
# 真實 PDF 的題目短標籤與題型分組（順序和 tests_car.py / tests_rp2040.py 一樣）
CAR_LABELS = [
    "Fuel octane", "Fuel tank capacity", "Engine displacement", "Oil incl. filter", "Oil viscosity",
    "Tire size", "Spare tire pressure", "Spare tire speed limit", "Key fob battery", "Cruise control min speed",
    "Tread wear indicator", "Max tire age", "Overall length", "A/C refrigerant", "Brake fluid spec",
    "Child in front seat", "Dead battery", "Overheating", "Fuel-saving button", "Jump-start voltage",
    "CVT fluid capacity", "Coolant capacity", "Overall width", "Halogen bulb wattage",
    "Oil without filter", "Spare wheel size", "Wheelbase",
    "Low oil pressure light", "When to calibrate TPMS", "Antifreeze ratio", "Monthly pressure loss",
    "Charging system light",
    "Price", "Max horsepower", "0-100 km/h time", "New-car warranty",
]
CAR_GROUPS = QUESTION_GROUPS
RP2040_LABELS = [
    "Max clock", "SRAM size", "Max external flash", "GPIO count", "ADC bits & rate",
    "VREG default voltage", "VREG max current", "USB boot crystal", "Temp sensor Vbe @27°C", "ADC base address",
    "XOSC frequency range", "Reflow peak temp", "Package type", "Storage temp & humidity", "USB series resistor",
    "Solder liquidus temp", "ADC max input @1.8V IO", "Lowest-power mode", "Wake-up time (ROSC)",
    "Temp sensor current",
    "GPIO pull-up value", "VSEL 1111 voltage", "BOD default threshold", "USB_VDD minimum", "ROK threshold",
    "DORMANT DVDD current", "ADC ENOB",
    "XOSC base address", "VREG_AND_CHIP_RESET base", "ADC low-supply voltage",
    "Core voltage @200MHz", "ADC via DMA bits",
    "Price", "Bluetooth", "MTBF", "Warranty",
]
RP2040_GROUPS = [
    ("Basic", 1, 15), ("Paraphrase", 16, 20), ("Table detail", 21, 27),
    ("Confusable", 28, 30), ("Multi-part", 31, 32), ("Trap (no answer)", 33, 36),
]
CORRECT_COLOR, WRONG_COLOR = "#cde2fb", "#eb6834"
# =====================================================


def load_file(filename):
    with open(RESULTS_DIR / filename, encoding="utf-8-sig") as f:
        return {row["name"]: row for row in csv.DictReader(f)}


def load(kind, result_files=RESULT_FILES, skip_parent=False):
    """讀取每種文件的某一類結果，回傳 {文件種類: {設定名稱: 那一列}}
    skip_parent：拿掉 parent 系列（真實 PDF 沒有章節，parent 等於把整份文件交出去，分數沒有意義）"""
    data = {}
    for series, files in result_files.items():
        rows = load_file(files[kind])
        if skip_parent:
            rows = {name: row for name, row in rows.items() if not row.get("parent")}
        data[series] = rows
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

    values = []
    for offset, (series, rows) in zip((-0.16, 0.16), data.items()):
        xs, ys = [], []
        for i, name in enumerate(names):
            v = score(rows[name][column]) if name in rows else None
            if v is not None:
                xs.append(v)
                ys.append(i + offset)
        values += xs
        ax.scatter(xs, ys, s=64, color=SERIES_COLORS[series], marker=SERIES_MARKERS.get(series, "o"),
                   edgecolors=SURFACE, linewidths=2, label=series, zorder=3)

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


def accuracy_vs_context(answers, filename="accuracy_vs_context.png", label_left=LABEL_LEFT,
                        merge_dist=40, merge_across=False):
    """回答正確率 vs. 每題交給 LLM 的字數：越靠左上越好（答得準又省字）。
    位置幾乎相同的點（同一列、x 差不到 merge_dist）合併成一個標籤，例如 baseline 和 parent-100。
    merge_across=True 時，不同文件的點也一起合併（兩份文件的 baseline 剛好疊在一起時用）。"""
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    groups = []
    for series, rows in answers.items():
        xs = [int(r["avg_context"]) for r in rows.values()]
        ys = [score(r["answer"]) for r in rows.values()]
        ax.scatter(xs, ys, s=72, color=SERIES_COLORS[series], marker=SERIES_MARKERS.get(series, "o"),
                   edgecolors=SURFACE, linewidths=2, label=series, zorder=3)
        if not merge_across:
            groups = []
        start = len(groups)
        for name, x, y in zip(rows, xs, ys):
            for g in groups:
                if g["y"] == y and abs(g["x"] - x) < merge_dist:
                    if name not in g["names"]:
                        g["names"].append(name)
                    break
            else:
                groups.append({"x": x, "y": y, "names": [name], "series": series})
        if not merge_across:
            draw_labels(ax, groups[start:], label_left)
    if merge_across:
        draw_labels(ax, groups, label_left)

    ax.set_xlabel("Characters sent to the LLM per question (avg)", color=INK_2, fontsize=9)
    ax.set_ylabel(f"Answers correct (out of {N_ANSWER})", color=INK_2, fontsize=9)
    ax.set_xlim(0, None)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax, "Accuracy vs. context size", "Top-left is best: more correct answers with less context")
    legend(ax, loc="lower right")
    save(fig, filename)


def draw_labels(ax, groups, label_left):
    for g in groups:
        label = " / ".join(g["names"])
        to_left = (g["series"], label) in label_left
        ax.annotate(label, (g["x"], g["y"]), xytext=(-8 if to_left else 8, -3), textcoords="offset points",
                    fontsize=8, color=INK_2, ha="right" if to_left else "left")


def chunk_size_effect(retrieval, answers, filename="chunk_size_effect.png"):
    """依標題切（heading）時，每段字數上限對檢索與回答的影響。兩張小圖，都是百分比。"""
    sizes = {"heading-100": 100, "heading-200": 200, "heading-300": 300, "baseline": 600}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor=SURFACE, sharey=True)
    panels = [
        (axes[0], retrieval, "retrieval", N_RETRIEVAL, "Retrieval hit rate"),
        (axes[1], answers, "answer", N_ANSWER, "Answer accuracy"),
    ]
    notes = {"retrieval": "", "answer": "; 300 not run with the LLM (dashed)"}
    lowest = 100
    for ax, data, column, total, title in panels:
        for series, rows in data.items():
            # 沒有跑過的大小用 NaN，實線會斷開，不會假裝中間有資料；斷開的地方另外用虛線接起來，看得出趨勢
            ys = [100 * score(rows[n][column]) / total if n in rows and score(rows[n][column]) is not None else float("nan")
                  for n in sizes]
            measured = [(x, y) for x, y in zip(sizes.values(), ys) if y == y]  # y == y 排除 NaN
            lowest = min([lowest] + [y for _, y in measured])
            if len(measured) < len(ys):
                ax.plot(*zip(*measured), color=SERIES_COLORS[series], linewidth=1.2, linestyle=(0, (3, 3)), zorder=2)
            ax.plot(list(sizes.values()), ys, color=SERIES_COLORS[series], linewidth=2,
                    marker=SERIES_MARKERS.get(series, "o"), markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=series, zorder=3)
        ax.set_xticks(list(sizes.values()))
        ax.set_xlabel("Max characters per chunk", color=INK_2, fontsize=9)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax, title, f"heading strategy, out of {total} questions{notes[column]}")
    axes[0].set_ylabel("Correct (%)", color=INK_2, fontsize=9)
    axes[0].set_ylim(min(60, lowest // 10 * 10), 102)
    legend(axes[1], loc="lower right")
    save(fig, filename)


def per_question_heatmap(data, column, n_questions, title, subtitle, filename,
                         labels=QUESTION_LABELS, groups=QUESTION_GROUPS):
    """逐題熱力圖：每一列是一題、每一欄是一個設定，答錯（或沒命中）的格子標成橘色並打 ×。
    data 裡每種文件畫一張小圖（左右並排，題目要相同）；labels、groups 是題目短標籤與題型分組。"""
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    assert len(labels) >= n_questions, "labels 比題目少"
    seen = {name for rows in data.values() for name in rows}
    names = [c["name"] for c in exp.CONFIGS if c["name"] in seen]
    groups = [g for g in groups if g[1] <= n_questions]

    cell = 0.26
    fig, axes = plt.subplots(1, len(data), figsize=(len(data) * cell * len(names) + 4.8, cell * n_questions + 2.4),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (series, rows) in zip(axes, data.items()):
        grid = np.full((n_questions, len(names)), np.nan)
        totals = []
        for j, name in enumerate(names):
            if name not in rows or rows[name][column] in ("", None):
                totals.append("")
                continue
            failed = {int(q) for q in rows[name][column].split() if q.isdigit()}
            grid[:, j] = [0 if q in failed else 1 for q in range(1, n_questions + 1)]
            totals.append(str(n_questions - len(failed)))

        ax.pcolormesh(grid, cmap=ListedColormap([WRONG_COLOR, CORRECT_COLOR]), vmin=0, vmax=1,
                      edgecolors=SURFACE, linewidth=1.5)
        for q, j in zip(*np.where(grid == 0)):
            ax.text(j + 0.5, q + 0.5, "×", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        for q, j in zip(*np.where(np.isnan(grid))):
            ax.text(j + 0.5, q + 0.5, "–", ha="center", va="center", fontsize=8, color=MUTED)

        # 題型之間留一條較粗的空白
        for _, first, _ in groups[1:]:
            ax.axhline(first - 1, color=SURFACE, linewidth=5)
        # 欄底下寫每個設定的總分
        for j, total in enumerate(totals):
            ax.text(j + 0.5, n_questions + 0.6, total, ha="center", va="top", fontsize=8, color=INK)

        ax.set_ylim(n_questions, 0)
        ax.set_xticks([j + 0.5 for j in range(len(names))], names, rotation=55, ha="left", rotation_mode="anchor")
        ax.xaxis.tick_top()
        ax.set_yticks([q + 0.5 for q in range(n_questions)],
                      [f"Q{q + 1}  {labels[q]}" for q in range(n_questions)])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.set_xlabel(series, color=INK, fontsize=10, fontweight="bold", labelpad=22)

    # 題型名稱寫在右邊那張圖的右側
    for name, first, last in groups:
        axes[-1].text(len(names) + 0.4, (first - 1 + min(last, n_questions)) / 2, name,
                     va="center", fontsize=8, color=INK_2)

    fig.suptitle(title, x=0.02, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.02, 0.955 - 0.25 / fig.get_figheight(), subtitle, fontsize=9, color=INK_2)
    fig.legend(handles=[Patch(color=CORRECT_COLOR, label="Correct"), Patch(color=WRONG_COLOR, label="Wrong  (×)")],
               loc="lower right", frameon=False, fontsize=9, labelcolor=INK_2, ncol=2)
    fig.text(0.02, 0.01, "Numbers below each column = total correct for that config.", fontsize=8, color=MUTED)
    fig.subplots_adjust(top=1 - 2.0 / fig.get_figheight(), bottom=0.9 / fig.get_figheight(), wspace=0.08)
    save(fig, filename)


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
    per_question_heatmap(retrieval, "missed", N_RETRIEVAL, "Retrieval by question",
                         "Did the retrieved chunks contain the answer? (traps excluded)", "per_question_retrieval.png")
    if all((RESULTS_DIR / f).exists() for f in DETAIL_FILES.values()):
        details = {series: load_file(f) for series, f in DETAIL_FILES.items()}
        per_question_heatmap(details, "wrong", N_ANSWER, "LLM answers by question",
                             "Did qwen2.5:3b's answer contain every keyword? (separate rerun)", "per_question_answers.png")

    # ---------- 真實說明書 PDF（結果四） ----------
    real_retrieval = load("retrieval", REAL_FILES, skip_parent=True)
    real_answers = load("answer", REAL_FILES)
    dot_plot(real_retrieval, "retrieval", N_RETRIEVAL, "Real manuals: retrieval hit rate by config",
             "32 questions, traps excluded; parent configs omitted (no chapters, so they send the whole PDF)",
             "real_retrieval_hits.png", legend_loc="upper left")
    accuracy_vs_context(real_answers, "real_accuracy_vs_context.png", label_left=REAL_LABEL_LEFT,
                        merge_dist=60, merge_across=True)
    chunk_size_effect(real_retrieval, real_answers, "real_chunk_size_effect.png")
    for series, labels, groups, filename in [
        ("Car manual (zh)", CAR_LABELS, CAR_GROUPS, "real_per_question_answers_car.png"),
        ("RP2040 datasheet (en)", RP2040_LABELS, RP2040_GROUPS, "real_per_question_answers_rp2040.png"),
    ]:
        per_question_heatmap({series: real_answers[series]}, "wrong", N_ANSWER, f"LLM answers by question: {series}",
                             "Did qwen2.5:3b's answer contain every keyword?", filename, labels, groups)


if __name__ == "__main__":
    main()
