from scripts.run_main_h1 import main as run_h1
from scripts.run_main_h2 import main as run_h2
from scripts.plot_h1 import main as plot_h1
from scripts.plot_h2 import main as plot_h2


def main():
    print("H1: эксперимент с семантическими ловушками")
    run_h1()

    print("H2: эксперимент со сменой предпочтений")
    run_h2()

    print("Построение графиков")
    plot_h1()
    plot_h2()

    print("\nЭксперименты завершены.")
    print("Таблицы: runs/")
    print("Графики: figures/")


if __name__ == "__main__":
    main()