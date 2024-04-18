# -*- coding: utf-8 -*-
"""

@author: kirdr
"""

import pandas as pd
from datetime import datetime, timedelta

def convert_data(input_csv, output_csv):
    # Чтение входного файла CSV
    df = pd.read_csv(r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train.csv")

    # Преобразование данных
    reshaped_data = []
    labels = df.iloc[:, 0].tolist()  # Получаем список меток из первого столбца

    for i, label in enumerate(labels):
        current_time = datetime(1750, 1, 1)  # Базовая временная метка
        for j in range(1, len(df.columns)):  # Начинаем с 1, чтобы пропустить первый столбец с метками
            version = "V" + str(j)  # Формируем имя версии
            data = df.iloc[i, j]  # Получаем данные для текущей ячейки
            if not pd.isna(data):  # Проверяем, что данные не являются пустыми
                reshaped_data.append([label, current_time.strftime('%Y-%m-%d %H:%M:%S'), data])
            current_time += timedelta(hours=1)  # Увеличение временной метки на один час

    # Создание DataFrame для новых данных
    new_df = pd.DataFrame(reshaped_data, columns=['item_id', 'timestamp', 'target'])

    # Запись нового DataFrame в CSV файл
    new_df.to_csv(output_csv, index=False)

input_file = r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train.csv"
output_file = r"C:\Users\kirdr\.spyder-py3\TimeDataSets\Hourly-train-processed.csv"
convert_data(input_file, output_file)