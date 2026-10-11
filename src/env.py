from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class EnvConfig:
    """Конфигурация среды StreamingLambdaEnv."""

    mode: str = "paradox"  # aligned/paradox
    seed: int = 42
    horizon: int = 500  # длина эпизода

    # λ (дискретные бины от 0.0 до 1.0)
    lambda_bins: Tuple[float, ...] = tuple(np.round(np.linspace(0, 1, 11), 1))

    # Награда и пороги
    reward_noise_sigma: float = 0.0
    override_threshold: float = 0.35  # если |λ-λ*| > threshold => override=True

    # Параметры ловушек
    trap_prob: float = 0.4  # вероятность ловушки в режиме paradox

    # Параметры дрейфа предпочтений (Concept Drift, H2)
    use_drift: bool = False
    drift_step: int = 250
    drift_type: str = "swap"  # инверсия правил comfort/explore


class StreamingLambdaEnv:
    """
    Текстовая среда для управления параметром λ (баланс точность/разнообразие).
    Симулирует поток контекстов пользователя онлайн-кинотеатра.
    """

    def __init__(self, cfg: EnvConfig):
        assert cfg.mode in ("aligned", "paradox")
        assert len(cfg.lambda_bins) >= 2
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.t = 0
        self.current: Optional[Dict[str, Any]] = None

        # Маппинг правил до и после Concept Drift (t >= drift_step)
        self.map_pre = {"comfort": 0.9, "explore": 0.1}
        self.map_post = {"comfort": 0.1, "explore": 0.9}

        # Контексты для режима Aligned
        self.templates_aligned = {
            "comfort": [
                "Late evening. I'm alone. I want something familiar and relaxing.",
                "After work I'm tired. Give me a safe, similar pick.",
                "Quiet night. No experiments. Something close to my taste.",
                "Just want to rewatch something cozy today.",
                "Feeling exhausted. Need my usual comfort movies.",
                "Standard evening routine. Nothing new please.",
                "Focus mode. I need background movie I already know.",
                "Lazy Sunday. Stick to the classics I love.",
                "I don't want to think. Give me the most predictable choice.",
                "Stressful day. I need something familiar to calm down.",
            ],
            "explore": [
                "Weekend with friends. We want something new and surprising.",
                "I'm bored of the same. Show me diverse options.",
                "Morning energy. I want to discover something different.",
                "Surprise me! I'm in a bold mood for experiments.",
                "Feeling adventurous. Give me a random genre mix.",
                "Party time. Need fresh hits and new discoveries.",
                "Tired of routine. Show me what else is out there.",
                "Exploring new cultures today. Give me something exotic.",
                "Wide awake and curious. Show me the latest releases.",
                "Looking for inspiration in unusual places. Diversify!",
            ],
        }

        # Парные контексты-ловушки для режима Paradox
        self.trap_pairs = [
            ("Friday night. I'm alone.", "Friday night. With friends."),
            ("Watching a movie alone tonight.", "Watching a movie with friends tonight."),
            ("Quiet evening, just me.", "Quiet evening, but guests are here."),
            ("Dinner for one and a show.", "Dinner party for five and a show."),
            ("Looking for something to watch by myself.", "Looking for something to watch with the crew."),
            ("Late night movie. Feeling sleepy.", "Late night movie. Feeling energetic."),
            ("Just woke up, feeling drowsy.", "Just woke up, feeling sharp."),
            ("Midnight session. Energy is low.", "Midnight session. Energy is peaking."),
            ("End of the day. Barely awake.", "End of the day. Feeling restless."),
            ("Monday morning. Working from home.", "Monday morning. On a day off."),
            ("Background for my office tasks.", "Background for my holiday party."),
            ("Focusing on a project right now.", "Relaxing on my vacation right now."),
            ("Busy day ahead. Need a soundtrack.", "Free day ahead. Need a soundtrack."),
            ("Concentrated mood. I have a deadline.", "Carefree mood. I have no plans."),
            ("Watching on my big screen TV.", "Watching on my small smartphone."),
            ("Home cinema mode engaged.", "Commuting on a bus with my phone."),
            ("Living room setup, lights off.", "Train station waiting, lights bright."),
            ("Relaxing on the sofa with a remote.", "Walking in the park with my mobile."),
            ("Immersive experience requested.", "Quick snack-sized content requested."),
        ]

        self.rule_equivalence = {
            "explore": [
                "Surprise me with something unconventional.",
                "Give me a random pick outside my usual genres.",
                "I want to escape my routine. Something totally different.",
            ],
            "comfort": [
                "Rewatch mood. Similar vibe as last time.",
                "Something predictable, like my favorites.",
                "No surprises today. Keep it close to what I like.",
            ],
        }

    def _to_bin(self, action: Union[float, int]) -> float:
        """Приведение действия к ближайшему дискретному значению λ."""
        bins = np.array(self.cfg.lambda_bins, dtype=float)
        if isinstance(action, (int, np.integer)):
            idx = int(np.clip(action, 0, len(bins) - 1))
            return float(bins[idx])

        x = float(np.clip(float(action), 0.0, 1.0))
        idx = int(np.argmin(np.abs(bins - x)))
        return float(bins[idx])

    def _intent_to_lambda_star(self, intent: str) -> float:
        """Идеальное значение λ* с учетом вероятного Concept Drift."""
        if self.cfg.use_drift and self.t >= self.cfg.drift_step:
            if self.cfg.drift_type == "swap":
                return float(self.map_post[intent])
        return float(self.map_pre[intent])

    def _sample_aligned(self) -> Dict[str, Any]:
        """Генерация контекста для режима Aligned."""
        intent = "comfort" if (self.rng.random() < 0.5) else "explore"
        text = str(self.rng.choice(self.templates_aligned[intent]))
        lam_star = self._to_bin(self._intent_to_lambda_star(intent))
        return {
            "context_text": text,
            "intent": intent,
            "lambda_star": lam_star,
            "is_trap": False,
            "trap_id": None,
        }

    def _sample_paradox(self) -> Dict[str, Any]:
        """Генерация контекста для режима Paradox (ловушки)."""
        if self.rng.random() < self.cfg.trap_prob:
            trap_id = int(self.rng.integers(0, len(self.trap_pairs)))
            a, b = self.trap_pairs[trap_id]
            side = int(self.rng.integers(0, 2))
            text = a if side == 0 else b
            intent = "comfort" if side == 0 else "explore"

            lam_star = self._to_bin(self._intent_to_lambda_star(intent))
            return {
                "context_text": text,
                "intent": intent,
                "lambda_star": lam_star,
                "is_trap": True,
                "trap_id": trap_id,
            }

        intent = "comfort" if (self.rng.random() < 0.5) else "explore"
        text = str(self.rng.choice(self.rule_equivalence[intent]))
        lam_star = self._to_bin(self._intent_to_lambda_star(intent))
        return {
            "context_text": text,
            "intent": intent,
            "lambda_star": lam_star,
            "is_trap": False,
            "trap_id": None,
        }

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        """Сброс среды в начальное состояние."""
        self.t = 0
        self.current = (
            self._sample_aligned()
            if self.cfg.mode == "aligned"
            else self._sample_paradox()
        )
        obs = self.current["context_text"]
        info = {
            k: self.current[k]
            for k in ("intent", "lambda_star", "is_trap", "trap_id")
        }
        info["t"] = self.t
        return obs, info

    def step(self, action: Union[float, int]) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Обработка действия агента."""
        assert self.current is not None, "Call reset() first."

        prev = self.current
        lam_pred = self._to_bin(action)
        lam_star = float(prev["lambda_star"])

        reward = 1.0 - abs(lam_pred - lam_star)
        if self.cfg.reward_noise_sigma > 0:
            reward += float(self.rng.normal(0.0, self.cfg.reward_noise_sigma))
        reward = float(np.clip(reward, 0.0, 1.0))

        override = abs(lam_pred - lam_star) > self.cfg.override_threshold

        self.t += 1
        done = self.t >= self.cfg.horizon

        if not done:
            self.current = (
                self._sample_aligned()
                if self.cfg.mode == "aligned"
                else self._sample_paradox()
            )
            next_obs = self.current["context_text"]
            next_intent = self.current["intent"]
            next_is_trap = self.current["is_trap"]
            next_trap_id = self.current["trap_id"]
        else:
            next_obs = ""
            next_intent = None
            next_is_trap = None
            next_trap_id = None

        info = {
            "t": self.t,
            "mode": self.cfg.mode,
            "lambda_pred": lam_pred,
            "reward": reward,
            "override": override,
            "prev_intent": prev["intent"],
            "prev_lambda_star": lam_star,
            "prev_is_trap": prev["is_trap"],
            "prev_trap_id": prev["trap_id"],
            "next_intent": next_intent,
            "next_is_trap": next_is_trap,
            "next_trap_id": next_trap_id,
        }

        return next_obs, reward, done, info