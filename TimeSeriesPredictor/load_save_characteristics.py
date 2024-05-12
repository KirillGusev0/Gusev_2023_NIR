# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:03:03 2024

@author: kirdr
"""
import pandas as pd
import os

DATA_FILE = r"C:/Users/kirdr/.spyder-py3/TimeSeriesPredictor/Data-container.csv"

def save_time_series_characteristics(time_series_characteristics, data_file):
    try:
        # Вывод значений словаря перед сохранением
        print("Характеристики временного ряда перед сохранением:")
        for key, value in time_series_characteristics.items():
            print(f"{key}: {value}")

        # Создаем DataFrame только с новыми характеристиками
        new_df = pd.DataFrame([time_series_characteristics])

        # Проверяем существование файла
        if not os.path.exists(data_file):
            # Если файл не существует, просто сохраняем новый DataFrame
            new_df.to_csv(data_file, index=False)
        else:
            # Если файл существует, загружаем его
            existing_df = pd.read_csv(data_file)
            # Объединяем существующий DataFrame с новым DataFrame
            df = pd.concat([existing_df, new_df], ignore_index=True)
            df.to_csv(data_file, index=False)

        print("Характеристики временного ряда успешно сохранены.")
    except Exception as e:
        print(f"Произошла ошибка при сохранении данных: {str(e)}")
        
    

def load_time_series_characteristics(time_series_index):
    try:
        # Загрузка данных из CSV файла в DataFrame
        df = pd.read_csv(DATA_FILE)
        # Получение характеристик временного ряда по его индексу
        time_series_characteristics = df[df['TimeSeriesIndex'] == time_series_index].to_dict(orient='list')
        if not time_series_characteristics:
            print(f"Ошибка: Характеристики временного ряда с индексом {time_series_index} не найдены.")
            return None
        return time_series_characteristics
    except FileNotFoundError:
        print("Ошибка: Файл данных не найден.")
        return None

