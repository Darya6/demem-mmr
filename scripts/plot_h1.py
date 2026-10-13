# scripts/plot_h1.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    path = os.path.join(RUNS_DIR, "results_H1.csv")
    if not os.path.exists(path):
        print(f"Файл {path} не найден. График не может быть построен.")
        return

    df = pd.read_csv(path)
    df = df[df["agent"].isin(["semantic", "decision"])].copy()
    df["K"] = pd.to_numeric(df["K"])
    df = df.sort_values("K")

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 4.5))

    sns.lineplot(
        data=df,
        x="K",
        y="mean_reward_trap",
        hue="agent",
        hue_order=["semantic", "decision"],
        palette={"semantic": "blue", "decision": "red"},
        marker="o",
        linewidth=2.5,
        errorbar=("ci", 95),
    )

    plt.axhline(0.6, color="black", linestyle="--", alpha=0.6, label="Без памяти (λ=0.5)")
    plt.title("Качество на семантических ловушках (trap_prob=0.4)")
    plt.xlabel("Объём памяти (K)")
    plt.ylabel("Средняя награда на ловушках")
    plt.xticks([3, 5, 10, 20])

    handles, labels = plt.gca().get_legend_handles_labels()
    label_map = {"semantic": "Семантическая память", "decision": "Селективная память"}
    labels = [label_map.get(x, x) for x in labels]
    plt.legend(handles, labels, title="Агент", loc="best")

    out_path = os.path.join(FIGURES_DIR, "figure_h1_traps_trap04.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранён: {out_path}")


if __name__ == "__main__":
    main()