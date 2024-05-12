# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:41:01 2024

@author: kirdr
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

DATA_FILE = r"C:/Users/kirdr/.spyder-py3/TimeSeriesPredictor/Data-container.csv"

def analyze_time_series_from_csv(csv_file):
    # Чтение файла существующих данных, если он существует
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # Определение максимального значения индекса временного ряда
        max_index = df['TimeSeriesIndex'].max()
        # Если поле TimeSeriesIndex пустое, то записываем 1, иначе максимальное значение + 1
        time_series_index = 1 if pd.isnull(max_index) else max_index + 1
    else:
        # Если файла нет, начинаем с 1
        time_series_index = 1
    
    # Загрузка данных из CSV файла
    time_series = pd.read_csv(csv_file)
    time_series['timestamp'] = pd.to_datetime(time_series['timestamp'])
    # Удаляем столбец item_id перед вычислением среднего значения временных меток
    time_series = time_series.drop(columns=['item_id'])
    # Удаляем столбец временных меток перед вычислением дисперсии
    time_series = time_series.drop(columns=['timestamp'])
    
    # Определение статистических свойств временного ряда
    variance = time_series['target'].var()
    
    # Определение периодичности временного ряда
    autocorrelation = sm.tsa.acf(time_series['target'], nlags=1)[1]
    is_periodic = abs(autocorrelation) > 0.5
    
    # Определение пространственной структуры временного ряда
    num_dimensions = time_series.shape[1] - 1  # Вычитаем один, так как один столбец обычно содержит временную метку
    
    # Определение наличия тренда и сезонности
    trend = "with trend" if abs(np.polyfit(range(len(time_series)), time_series['target'], 1)[0]) > 0.1 else "without trend"
    seasonal = "with seasonality" if is_periodic else "without seasonality"
    
    # Определение типа шума
    noise_type = "with autoregressive noise" if abs(autocorrelation) > 0.2 else "with white noise"
    
    # Определение размера выборки
    num_samples = len(time_series)
    sample_size = "long" if num_samples > 1000 else "short"
    
    # Формирование словаря с характеристиками временного ряда
    time_series_characteristics = {
        "TimeSeriesIndex": time_series_index,
        "TimeSeriesPath": csv_file,
        "Variance": variance,
        "IsPeriodic": is_periodic,
        "NumDimensions": num_dimensions,
        "Trend": trend,
        "Seasonal": seasonal,
        "NoiseType": noise_type,
        "SampleSize": sample_size
    }
    print("Словарь time_series_characteristics в analyze_time_series_from_csv:", time_series_characteristics)
    return time_series_characteristics