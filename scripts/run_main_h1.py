import os
import pandas as pd
from src.runner import run_grid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")


def summarize_and_save(df, filename_prefix, out_dir=RUNS_DIR):
    metrics = ["mean_reward", "mean_reward_trap", "write_rate"]
    summary = (
        df.groupby(["agent", "K"])[metrics]
          .agg(["mean", "std"])
          .reset_index()
    )

    for m in metrics:
        summary[f"{m}_final"] = summary.apply(
            lambda x: f"{x[(m, 'mean')]:.3f} ± {(0.0 if pd.isna(x[(m, 'std')]) else x[(m, 'std')]):.3f}",
            axis=1
        )

    report_cols = ["agent", "K"] + [f"{m}_final" for m in metrics]
    final_table = summary[report_cols]
    final_table.columns = [c[0] if isinstance(c, tuple) else c for c in final_table.columns]

    print(f"\nРезультат для {filename_prefix}")
    print(final_table.to_string(index=False))

    os.makedirs(out_dir, exist_ok=True)
    final_table.to_csv(os.path.join(out_dir, f"summary_{filename_prefix}.csv"), index=False)
    return final_table


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    res_df = run_grid(
        Ks=(3, 5, 10, 20),
        seeds=range(10),
        horizon=500,
        trap_prob=0.4,
        mode="paradox",
        k_retrieve=3,
    )
    res_df.to_csv(os.path.join(RUNS_DIR, "results_H1.csv"), index=False)
    summarize_and_save(res_df, "h1_trap04_paradox", RUNS_DIR)


if __name__ == "__main__":
    main()