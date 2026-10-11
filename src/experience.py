from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Experience:
    """Класс для хранения единицы опыта агента."""
    context: str
    target_lambda: float   # Идеальное значение лямбда
    reward: float          # Полученная награда
    override: bool         # Флаг ошибки
    t: int                 # Временной шаг
    emb: Optional[np.ndarray] = None    # Эмбеддинг
