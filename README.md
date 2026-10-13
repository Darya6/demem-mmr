# Селективная память для адаптивного выбора параметра λ в MMR

Проект воспроизводит эксперименты для статьи **«Селективная память для адаптивного выбора параметра λ в методе MMR»**. Исследуется, как отбор записей в памяти влияет на выбор λ в синтетической потоковой среде.

Сравниваются три агента:
- **NoMemory** — агент без памяти с фиксированным λ = 0,5;
- **Semantic Memory** — сохраняет все наблюдения и использует семантический поиск;
- **Selective Memory** — сохраняет наблюдения, для которых награда ниже заданного порога.

Для семантического поиска используется модель `all-MiniLM-L6-v2` из Sentence Transformers, косинусная близость и извлечение трёх ближайших записей.


## Быстрый запуск

### Вариант 1. Google Colab

```
!git clone https://github.com/Darya6/selective-memory-MMR.git
%cd selective-memory-MMR
!pip install -r requirements.txt
!python main.py
```

`main.py` последовательно запускает H1 и H2, сохраняет CSV-таблицы и строит графики. Время выполнения зависит от среды и загрузки модели Sentence Transformers при первом запуске.

Чтобы вывести сводные таблицы после завершения:

```
import pandas as pd
from IPython.display import display

display(pd.read_csv("runs/summary_h1_trap04_paradox.csv"))
display(pd.read_csv("runs/summary_h2_drift_trap04.csv"))
```

Чтобы отобразить построенные графики:

```
from IPython.display import Image, display

display(Image(filename="figures/figure_h1_traps_trap04.png"))
display(Image(filename="figures/figure_h2_recovery_time.png"))
```

Чтобы скачать результаты:

```
from google.colab import files

for path in [
    "runs/results_H1.csv",
    "runs/summary_h1_trap04_paradox.csv",
    "runs/results_H2.csv",
    "runs/summary_h2_drift_trap04.csv",
    "figures/figure_h1_traps_trap04.png",
    "figures/figure_h2_recovery_time.png",
]:
    files.download(path)
```

### Вариант 2. PyCharm или локальный терминал

Требуется Python 3.10 или 3.11.

1. Откройте папку проекта в PyCharm.
2. Создайте виртуальное окружение.
3. В терминале выполните:

```
python -m pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

После выполнения файлы появятся в `runs/` и `figures/`.

## Параметры экспериментов

- Длина эпизода: 500 шагов.
- Случайные seed: 10 (от 0 до 9).
- Бюджеты памяти: `K ∈ {3, 5, 10, 20}`.
- Вероятность семантической ловушки: `trap_prob = 0.4`.
- Число извлекаемых записей: `k_retrieve = 3`.
- Порог записи для Selective Memory: награда ниже `0.8`.
- В H2 смена предпочтений происходит на шаге `t = 250`.
- Время восстановления измеряется как число шагов после смены предпочтений до достижения скользящей средней награды не ниже `0.8` на окне из 20 шагов. Если порог не достигнут, используется длина оставшейся части эпизода.

## Результаты

При успешном запуске создаются следующие файлы.

### Каталог `runs/`

- `results_H1.csv` — результаты отдельных запусков H1;
- `summary_h1_trap04_paradox.csv` — сводные показатели H1 по агентам и бюджетам памяти;
- `results_H2.csv` — результаты отдельных запусков H2;
- `summary_h2_drift_trap04.csv` — сводные показатели H2 по агентам и бюджетам памяти.

### Каталог `figures/`

- `figure_h1_traps_trap04.png` — награда на семантических ловушках;
- `figure_h2_recovery_time.png` — время восстановления после смены предпочтений.

В сводных таблицах представлены средние значения и стандартные отклонения по количеству запусков(seed=10). Результаты графиков должны соответствовать CSV, созданным текущим запуском.


