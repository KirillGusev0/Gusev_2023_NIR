# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:01:58 2024

@author: kirdr
"""
import time
from datetime import datetime
from load_save_characteristics import save_time_series_characteristics, load_time_series_characteristics
from time_series_analyzer import analyze_time_series_from_csv
from parametrs_list_and_choose import parametrs_choose, calculate_time
from time_series_predictor import predictions_with_chosen_parametrs
# Путь к файлу данных
DATA_FILE = r"C:/Users/kirdr/.spyder-py3/TimeSeriesPredictor/Data-container.csv"

def main():
    # Получаем локальное время начала программы
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Локальное время начала программы: {start_time}")

    # Записываем время начала выполнения программы
    start_execution_time = time.time()
    
    while True:
        # Предложение пользователю выбрать действие
        print("Выберите действие:")
        print("1. Классифицировать файл")
        print("2. Загрузить файл и продолжить прогнозирование")
        print("3. Выйти из программы")
        choice = input("Введите номер действия (1, 2 или 3): ")
        
        if choice == '1':
            # Классификация файла
            classify_file()
        elif choice == '2':
            # Загрузка файла и прогнозирование
            forecast_file()
        elif choice == '3':
            # Выйти из программы
            break
        else:
            print("Неверный выбор. Попробуйте снова.")

    # Записываем время окончания выполнения программы
    end_execution_time = time.time()
    # Вычисляем общее время выполнения программы
    total_execution_time = end_execution_time - start_execution_time
    print(f"Общее время выполнения программы: {total_execution_time:.2f} секунд")


def classify_file():
    # Получение пути к файлу временного ряда
    time_series_path = input(r"Введите путь к файлу временного ряда: ")
    # Анализ временного ряда
    time_series_characteristics = analyze_time_series_from_csv(time_series_path)
    if time_series_characteristics:
        # Сохранение характеристик временного ряда в файл данных
        save_time_series_characteristics(time_series_characteristics, DATA_FILE)
        print("Характеристики временного ряда успешно сохранены.")
    else:
        print("Ошибка при классификации временного ряда.")


def forecast_file():
    # Загрузка характеристик временного ряда
    time_series_index = int(input("Введите индекс временного ряда: "))
    loaded_time_series_characteristics = load_time_series_characteristics(time_series_index)

    if loaded_time_series_characteristics:
        print("Загруженные характеристики временного ряда:")
        print(loaded_time_series_characteristics)
        
        settings=parametrs_choose(loaded_time_series_characteristics)
        print(settings)
        file_path = loaded_time_series_characteristics['TimeSeriesPath'][0]
        file_path = file_path.strip("[]")           # Удаляем квадратные скобки
        file_path = file_path.strip("'") 
        # Прогнозирование временного ряда
        predictions_with_chosen_parametrs(file_path,settings)
       
    else:
        print("Ошибка при загрузке характеристик временного ряда.")


if __name__ == "__main__":
    main()