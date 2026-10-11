import pandas as pd
from sentence_transformers import SentenceTransformer

from src.env import EnvConfig, StreamingLambdaEnv
from src.agents import NoMemoryAgent, SimpleRetrievalAgent
from src.memory import SemanticMemory, DecisionMemory
from src.metrics import compute_metrics, compute_recovery_time


def run_episode(env, agent, max_steps: int = None):
    """Запуск одного эпизода агента в среде."""
    obs, _ = env.reset()
    logs = []
    done = False
    step = 0
    T = max_steps or env.cfg.horizon

    while not done and step < T:
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.observe(obs, reward, info)

        mem_size = len(agent.memory) if hasattr(agent, "memory") else 0
        mem_writes = getattr(agent.memory, "n_writes", 0) if hasattr(agent, "memory") else 0

        logs.append({
            "step": step,
            "context": obs,
            "action": action,
            "mem_size": mem_size,
            "mem_writes": mem_writes,
            **info
        })
        obs = next_obs
        step += 1

    return pd.DataFrame(logs)


def run_grid(
    Ks=(3, 5, 10, 20),
    seeds=range(10),
    horizon=500,
    trap_prob=0.4,
    mode="paradox",
    k_retrieve=3,
):
    """
    H1: mode=paradox и trap_prob=0.4.
    """
    shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    for K in Ks:
        for seed in seeds:
            cfg = EnvConfig(mode=mode, seed=seed, horizon=horizon, trap_prob=trap_prob)

            agent_factories = {
                "no_memory": lambda: NoMemoryAgent(),
                "semantic": lambda: SimpleRetrievalAgent(
                    SemanticMemory(K, encoder=shared_encoder), cfg.lambda_bins, k_retrieve
                ),
                "decision": lambda: SimpleRetrievalAgent(
                    DecisionMemory(K, encoder=shared_encoder), cfg.lambda_bins, k_retrieve
                ),
            }

            for name, make_agent in agent_factories.items():
                df = run_episode(StreamingLambdaEnv(cfg), make_agent())
                m = compute_metrics(df)
                m.update({
                    "agent": name,
                    "seed": seed,
                    "K": K,
                    "mode": mode,
                    "trap_prob": trap_prob,
                    "horizon": horizon,
                    "k_retrieve": k_retrieve,
                })
                rows.append(m)

    return pd.DataFrame(rows)


def run_grid_drift(
    Ks=(3, 5, 10, 20),
    seeds=range(10),
    trap_prob=0.4,
    horizon=500,
    drift_step=250,
    k_retrieve=3,
    store_if_reward_below=0.8,
    window=20,
    threshold=0.8,
):
    """H2: drift в режиме paradox, trap_prob=0.4."""
    shared_encoder = SentenceTransformer("all-MiniLM-L6-v2")

    rows = []
    for K in Ks:
        for seed in seeds:
            cfg = EnvConfig(
                mode="paradox",
                seed=seed,
                horizon=horizon,
                trap_prob=trap_prob,
                use_drift=True,
                drift_step=drift_step,
                drift_type="swap",
            )

            agent_factories = {
                "no_memory": lambda: NoMemoryAgent(default_lambda=0.5),
                "semantic": lambda: SimpleRetrievalAgent(
                    SemanticMemory(capacity=K, encoder=shared_encoder),
                    cfg.lambda_bins,
                    k_retrieve=k_retrieve,
                ),
                "decision": lambda: SimpleRetrievalAgent(
                    DecisionMemory(
                        capacity=K,
                        encoder=shared_encoder,
                        store_if_reward_below=store_if_reward_below
                    ),
                    cfg.lambda_bins,
                    k_retrieve=k_retrieve,
                ),
            }

            for agent_name, make_agent in agent_factories.items():
                env = StreamingLambdaEnv(cfg)
                agent = make_agent()
                df = run_episode(env, agent, max_steps=horizon)

                m = compute_metrics(df)
                m["recovery_time"] = compute_recovery_time(
                    df, drift_step=cfg.drift_step, window=window, threshold=threshold
                )
                m.update({
                    "agent": agent_name,
                    "seed": seed,
                    "K": K,
                    "trap_prob": trap_prob,
                    "horizon": horizon,
                    "drift_step": drift_step,
                    "k_retrieve": k_retrieve,
                    "window": window,
                    "threshold": threshold,
                })
                rows.append(m)

    return pd.DataFrame(rows)