from typing import Sequence
from collections import Counter
import numpy as np

from src.experience import Experience
from src.memory import BaseMemory


class NoMemoryAgent:
    """
    Baseline. Агент без памяти.
    Возвращает значение по умолчанию (λ = 0.5).
    """
    def __init__(self, default_lambda=0.5):
        self.default_lambda = float(default_lambda)

    def act(self, context_text: str) -> float:
        return self.default_lambda

    def observe(self, context_text: str, reward: float, info: dict) -> None:
        pass


class SimpleRetrievalAgent:
    """
    Агент с мажоритарным голосованием по памяти.
    Находит k похожих ситуаций и выбирает наиболее часто встречающееся решение.
    """
    def __init__(
        self,
        memory: BaseMemory,
        lambda_bins: Sequence[float],
        k_retrieve: int = 3,
        default_lambda: float = 0.5
    ):
        self.memory = memory
        self.lambda_bins = np.array(lambda_bins, dtype=float)
        self.k_retrieve = int(k_retrieve)
        self.default_lambda = float(default_lambda)

    def _to_bin(self, x: float) -> float:
        idx = int(np.argmin(np.abs(self.lambda_bins - float(x))))
        return float(self.lambda_bins[idx])

    def act(self, context_text: str) -> float:
        demos = self.memory.retrieve(context_text, k=self.k_retrieve)
        if not demos:
            return self._to_bin(self.default_lambda)

        targets = np.array([d.target_lambda for d in demos], dtype=float)
        lam = Counter(targets).most_common(1)[0][0]
        return self._to_bin(lam)

    def observe(self, context_text: str, reward: float, info: dict) -> None:
        exp = Experience(
            context=context_text,
            target_lambda=float(info["prev_lambda_star"]),
            reward=float(reward),
            override=bool(info["override"]),
            t=int(info["t"]),
        )
        self.memory.add(exp)