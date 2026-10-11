import pandas as pd


def compute_metrics(df: pd.DataFrame) -> dict:
    out = {}

    out["mean_reward"] = float(df["reward"].mean()) if len(df) else float("nan")

    if "prev_is_trap" in df.columns:
        trap_df = df[df["prev_is_trap"] == True]
        out["mean_reward_trap"] = float(trap_df["reward"].mean()) if len(trap_df) else float("nan")
    else:
        out["mean_reward_trap"] = float("nan")

    if "mem_writes" in df.columns and len(df):
        total_writes = int(df["mem_writes"].iloc[-1])
        out["total_writes"] = total_writes
        out["write_rate"] = float(total_writes / len(df))
    else:
        out["total_writes"] = 0
        out["write_rate"] = 0.0

    return out


def compute_recovery_time(
    df: pd.DataFrame,
    drift_step: int,
    window: int = 20,
    threshold: float = 0.8
) -> int:
    """
    recovery_time: шаги после drift_step до момента, когда rolling mean(reward) впервые >= threshold.
    """
    if len(df) == 0:
        return 0

    post = df[df["step"] >= drift_step].copy().reset_index(drop=True)

    if len(post) < window:
        return len(post)

    post["roll"] = post["reward"].rolling(window=window, min_periods=window).mean()
    hit = post.index[post["roll"] >= threshold]

    if len(hit) == 0:
        return len(post)

    return int(hit[0] + 1)