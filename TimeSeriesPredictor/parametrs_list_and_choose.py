

import pandas as pd
import os


def calculate_time(file_size_bytes):
    # Вычисление времени в секундах
    time_seconds = file_size_bytes / (1024 * 1024) * 20
    
   
    # Проверка на минимальное и максимальное значение
    if time_seconds > 43200:
        return 43200
    elif time_seconds < 1800:
        return 1800
    else:
        return time_seconds
    
def parametrs_choose(data_list):
   
    # словари с параметрами
    mode_dict = {
    "fast_training": "fast_training",
    "medium_quality": "medium_quality",
    "high_quality": "high_quality",
    "best_quality": "best_quality"
    }
    
    metrics_dict = {
    "SQL": "SQL",
    "WQL": "WQL",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "MASE": "MASE",
    "MSE": "MSE",
    "RMSE": "RMSE",
    "RMSLE": "RMSLE",
    "RMSSE": "RMSSE",
    "SMAPE": "SMAPE",
    "WAPE": "WAPE"
    }

    models_dict = {
    "Naive": "Naive",
    "SeasonalNaive": "SeasonalNaive",
    "ETS": "ETS",
    "Theta": "Theta",
    "RecursiveTabular": "RecursiveTabular",
    "DirectTabular": "DirectTabular",
    "TemporalFusionTransformer": "TemporalFusionTransformer"
    }
    
    
    # Задание длины прогноза
    prediction_length = int(input("Введите дальность прогноза: "))
    # Загрузка данных из CSV файла
    file_path = data_list['TimeSeriesPath'][0]
    file_path = file_path.strip("[]")           # Удаляем квадратные скобки
    file_path = file_path.strip("'") 
    time_series = pd.read_csv(file_path)
    time_series['timestamp'] = pd.to_datetime(time_series['timestamp'])
    # Удаляем столбец item_id перед вычислением среднего значения временных меток
    time_series = time_series.drop(columns=['item_id'])
    # Удаляем столбец временных меток перед вычислением дисперсии
    time_series = time_series.drop(columns=['timestamp'])
    # Выбор столбца временного ряда
    time_series_column = "target" 
    # Расчет среднего значения временного ряда
    mean_value = time_series[time_series_column].mean()
    
    #вычисление лимита времени
    file_size = os.path.getsize(file_path)
    time_limit=calculate_time(file_size)
    
    multiplaer=3
   
    
       
    
    enable_ensemble=False
    presets=mode_dict["medium_quality"]
    num_val_windows= 3
    metric=metrics_dict["MSE"]
    val_step_size = 2
    selected_models = {
        "DirectTabular": models_dict["DirectTabular"],
        "TemporalFusionTransformer": models_dict["TemporalFusionTransformer"]
        }
    
    
    if (data_list['Trend'] == "with trend")and(data_list['Seasonal'] == "with seasonality"):
        selected_models = {
            "Theta": models_dict["Theta"],
            "TemporalFusionTransformer": models_dict["TemporalFusionTransformer"]
            }
         
    if(data_list['SampleSize'] == "short"):
        print(prediction_length*num_val_windows)
        num_val_windows= 1
        val_step_size = 1 
        
    if(data_list['Variance'] < multiplaer*mean_value)and (data_list['NoiseType'] == "with autoregressive noise"):
       metric=metrics_dict["MAE"]
    elif(data_list['Seasonal'] == "with seasonality")and(data_list['Variance'] < multiplaer*mean_value):
        metric=metrics_dict["RMSSE"]
        if(data_list['NumDimensions'] == 0.0):
            metric=metrics_dict["MASE"]
    elif(data_list['Seasonal'] == "without seasonality")and(data_list['Trend'] == "without trend"):
        metric=metrics_dict["RMSLE"]
    if(val_step_size==1)and(num_val_windows==1):
        presets=mode_dict["high_quality"]
    # Формирование словаря с выбранными параметрами
    # Подготовка структуры hyperparameters на основе словаря selected_models
    hyperparameters = {model: {} for model in selected_models.keys()}
    list_of_parameters = {
        "prediction_length": prediction_length,
        "presets": presets,
        "metrics": metric,
        "enable_ensemble": enable_ensemble,
        "num_val_windows": num_val_windows,
        "time_limit": time_limit,
        "val_step_size": val_step_size,
        "models": selected_models,
        "hyperparameters": hyperparameters
    }
    return list_of_parameters

    