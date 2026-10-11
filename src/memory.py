from __future__ import annotations

from src.experience import Experience

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

class BaseMemory(ABC):
    """"Базовый интерфейс памяти."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.n_writes = 0

    @abstractmethod
    def add(self, exp: Experience) -> None:
        """Добавить один эпизод в память."""
        ...

    @abstractmethod
    def retrieve(self, query_text: str, k: int) -> List[Experience]:
        """Вернуть k наиболее релевантных записей из памяти."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Текущий размер памяти."""
        ...


class NoMemory(BaseMemory):
    """Baseline без памяти: ничего не хранит, ничего не возвращает."""
    def __init__(self, capacity: int = 0):
        super().__init__(capacity)

    def add(self, exp: Experience) -> None:
        return  # n_writes остаётся 0

    def retrieve(self, query_text: str, k: int) -> List[Experience]:
        return []

    def __len__(self) -> int:
        return 0


class SemanticMemory(BaseMemory):
    """
    Семантическая память.
    Использует эмбеддинги для поиска наиболее похожих по тексту ситуаций.
    """
    def __init__(self, capacity: int, encoder: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2"):
        super().__init__(capacity)
        if isinstance(encoder, str):
            self.encoder = SentenceTransformer(encoder)
        else:
            self.encoder = encoder

        self.buf = []
        self._cache = {}

    def _encode(self, text: str) -> np.ndarray:
        """"Превращает текст в эмбеддинг."""
        if text in self._cache:
            return self._cache[text]
        emb = self.encoder.encode(text, normalize_embeddings=True)
        self._cache[text] = emb
        return emb

    def add(self, exp: Experience) -> None:
        # Эмбеддинг ситуации для семантической памяти
        exp.emb = self._encode(exp.context)
        self.buf.append(exp)
        self.n_writes += 1
        if len(self.buf) > self.capacity:
            self.buf.pop(0)

    def retrieve(self, query_text: str, k: int):
        """
            Возвращает k записей, наиболее похожих на запрос по косинусной близости эмбеддингов.
            Если память пуста, возвращает пустой список.
            """
        if not self.buf:
            return []

        # Косинусное сходство между запросом и всеми записями
        q = self._encode(query_text)
        embs = np.stack([e.emb for e in self.buf], axis=0)
        sims = embs @ q # матричное умножение
        idx = np.argsort(-sims)[:k] # индексы k похожих записей
        return [self.buf[i] for i in idx]

    def __len__(self):
        return len(self.buf)


class DecisionMemory(SemanticMemory):
    """
    Память, ориентированная на решения.
    Сохраняет опыт только в том случае, если текущее решение агента
    было ошибочным или привело к низкой награде.
    """
    def __init__(self, capacity: int, encoder: Union[str, SentenceTransformer] = "all-MiniLM-L6-v2",
                 store_if_reward_below: float = 0.8):
        super().__init__(capacity, encoder=encoder)
        self.store_if_reward_below = store_if_reward_below

    def add(self, exp: Experience) -> None:
        """
            Сохраняет событие только если оно было критическим (override=True)
            или привело к низкой награде (reward < store_if_reward_below).
            """
        if not (exp.override or exp.reward < self.store_if_reward_below):
            return

        exp.emb = self._encode(exp.context)
        self.buf.append(exp)
        self.n_writes += 1

        if len(self.buf) > self.capacity:
            self.buf.pop(0)


