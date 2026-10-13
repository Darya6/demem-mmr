# scripts/plot_h2.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    path = os.path.join(RUNS_DIR, "results_H2.csv")
    if not os.path.exists(path):
        print(f"Файл {path} не найден. График не построен.")
        return

    df = pd.read_csv(path)
    df = df[df["agent"].isin(["semantic", "decision", "no_memory"])].copy()
    df["K"] = pd.to_numeric(df["K"])
    df = df.sort_values("K")

    palette = {"semantic": "blue", "decision": "red", "no_memory": "black"}
    linestyles = {"semantic": "-", "decision": "-", "no_memory": "--"}

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.5, 4.5))

    for agent in ["no_memory", "semantic", "decision"]:
        data_subset = df[df["agent"] == agent]
        sns.lineplot(
            data=data_subset,
            x="K",
            y="recovery_time",
            label={"no_memory": "Без памяти", "semantic": "Семантическая", "decision": "Селективная память"}[agent],
            marker="o",
            linewidth=2.5,
            color=palette[agent],
            linestyle=linestyles[agent],
            errorbar=("ci", 95),
        )

    plt.title("Время восстановления после смены предпочтений (trap_prob=0.4)")
    plt.xlabel("Объём памяти (K)")
    plt.ylabel("Шаги до восстановления")
    plt.xticks(sorted(df["K"].unique()))
    plt.legend(title="Агент", loc="best")
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "figure_h2_recovery_time.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"График сохранён: {out_path}")


if __name__ == "__main__":
    main()