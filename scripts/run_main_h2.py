import os
import pandas as pd
from src.runner import run_grid_drift

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(BASE_DIR, "runs")


def summarize_and_save_h2(df, filename_prefix, out_dir=RUNS_DIR):
    metrics = ["mean_reward", "recovery_time"]
    summary = (
        df.groupby(["agent", "K"])[metrics]
          .agg(["mean", "std"])
          .reset_index()
    )

    for m in metrics:
        summary[f"{m}_final"] = summary.apply(
            lambda x: f"{x[(m, 'mean')]:.2f} ± {(0.0 if pd.isna(x[(m, 'std')]) else x[(m, 'std')]):.2f}",
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

    HORIZON = 500
    DRIFT_STEP = 250

    res_df = run_grid_drift(
        Ks=(3, 5, 10, 20),
        seeds=range(10),
        trap_prob=0.4,
        horizon=HORIZON,
        drift_step=DRIFT_STEP,
        k_retrieve=3,
        store_if_reward_below=0.8,
        window=20,
        threshold=0.8,
    )

    res_df.to_csv(os.path.join(RUNS_DIR, "results_H2.csv"), index=False)
    summarize_and_save_h2(res_df, "h2_drift_trap04", RUNS_DIR)


if __name__ == "__main__":
    main()