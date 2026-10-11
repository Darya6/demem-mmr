from src.env import EnvConfig, StreamingLambdaEnv

def test_episode_length():
    cfg = EnvConfig(
        mode="paradox",
        seed=42,
        horizon=50,
        trap_prob=0.4,
    )

    env = StreamingLambdaEnv(cfg)
    obs, info = env.reset()

    steps = 0
    done = False

    while not done:
        obs, reward, done, info = env.step(0.5)
        steps += 1

    assert steps == cfg.horizon


def test_reward():
    cfg = EnvConfig(
        mode="aligned",
        seed=42,
        horizon=1,
    )

    env = StreamingLambdaEnv(cfg)
    env.reset()

    lambda_star = env.current["lambda_star"]

    _, reward, _, info = env.step(lambda_star)

    assert reward == 1.0
    assert info["lambda_pred"] == lambda_star


def test_drift_changes_mapping():
    cfg = EnvConfig(
        mode="aligned",
        seed=42,
        horizon=20,
        use_drift=True,
        drift_step=10,
    )

    env = StreamingLambdaEnv(cfg)

    assert env._intent_to_lambda_star("comfort") == 0.9

    env.t = 10

    assert env._intent_to_lambda_star("comfort") == 0.1


def test_no_memory_does_not_write():
    from src.memory import NoMemory
    from src.experience import Experience

    memory = NoMemory()

    exp = Experience(
        context="test",
        target_lambda=0.9,
        reward=0.2,
        override=True,
        t=0,
    )

    memory.add(exp)

    assert len(memory) == 0
    assert memory.n_writes == 0
    assert memory.retrieve("test", k=3) == []