# %%
# Setup Inicial
import mlflow
import pandas
import sklearn
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def read_data(name):
    df = pandas.read_csv(
        'data/' + name,
        delimiter=',',
        names = ['indice_tiempo', 'clasificacion_vuelo', 'pasajeros', 'asientos', 'vuelos'],
        skiprows = 1,
        parse_dates = ['indice_tiempo'],
        dtype = {
            'pasajeros': 'int32',
            'asientos': 'int32',
            'vuelos': 'int32'
        }
    )
    df['clasificacion_vuelo'] = df['clasificacion_vuelo'].astype('category')
    df = df.sort_values("indice_tiempo")
    return df

# %%
# Lectura dataset
dataset = read_data('vuelos_asientos_pasajeros.csv')
print(dataset.dtypes)
print(dataset.head())

# %%
# Experimento 1: Predecir la cantidad de pasajeros de vuelos internacionales
#
# ⚠️
# Por ahora dato fijo, una sola predicción para un día en particular
# Ejemplo: 31/03/2025
fecha_objetivo = pandas.to_datetime("2025-03-31")
# ⚠️
#
# Configuración MLFlow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pasajeros Internacionales")
with mlflow.start_run():
    data = dataset.copy()
    # Nos quedamos únicamente con los vuelos internacionales
    data = data[data['clasificacion_vuelo'] == 'Internacional']
    # Agregamos una columna con los días desde el inicio
    data['dias_desde_inicio'] = (data['indice_tiempo'] - data['indice_tiempo'].min()).dt.days
    # Split dinámico según fecha_objetivo para train y test
    train = data[data['indice_tiempo'] < fecha_objetivo]
    test = data[data['indice_tiempo'] == fecha_objetivo]
    X_train = train[['dias_desde_inicio']]
    X_train = X_train.astype('float64')  # Para no obtener el warning de NaN para enteros
    y_train = train['pasajeros']
    X_test = test[['dias_desde_inicio']]
    X_test = X_test.astype('float64')
    y_test = test['pasajeros']
    # Modelo, fit y predict
    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Log de parámetros y métricas
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("fecha_objetivo", fecha_objetivo.date().isoformat())
    mlflow.log_param("dias_desde_inicio_prediccion", int(X_test.iloc[0, 0]))
    mlflow.log_param("fecha_inicio", data['indice_tiempo'].min().date().isoformat())
    mlflow.log_metric("mae", mae)
    if len(y_test) > 1:
        score = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", score)
        print(f"R2: {score:.2f}")
    # Tags para filtrar en la web de mlflow
    mlflow.set_tags({
        "tag1": "valorTag1",
        "tag2": "valorTag2"
    })
    # Log del modelo
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    # input_example no influye en el entrenamiento ni en la predicción real,
    # es solo para documentación y validación
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_regresion_dia",
        input_example=input_example,
        signature=signature
    )
    # Registro del modelo
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/modelo_regresion_dia"
    result = mlflow.register_model(model_uri, "modelo_regresion_dia")
    # Impresión de resultados obtenidos
    print(f"Fecha real: {pandas.to_datetime(test['indice_tiempo'].values[0]).date()}")
    print(f"Pasajeros reales: {y_test.values[0]:,.0f}")
    print(f"Pasajeros predichos: {y_pred[0]:,.0f}")
    print(f"MAE: {mae:,.0f}")
    # Fecha real: 2025-03-31T00:00:00.000000000
    # Pasajeros reales: 44,042
    # Pasajeros predichos: 24,728
    # MAE: 19,314

# Experimento 2: Predicción de Pasajeros Internacionales con Features de Tiempo y Evaluación en el Futuro

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pasajeros Internacionales - Series de Tiempo")

# %%
with mlflow.start_run():
    data = dataset.copy()
    data = data[data['clasificacion_vuelo'] == 'Internacional']

    # 1. Feature Engineering
    # Extraemos información de la columna 'indice_tiempo'
    data['year'] = data['indice_tiempo'].dt.year
    data['month'] = data['indice_tiempo'].dt.month
    data['day_of_week'] = data['indice_tiempo'].dt.dayofweek # Lunes=0, Domingo=6
    data['day_of_year'] = data['indice_tiempo'].dt.dayofyear
    data['week_of_year'] = data['indice_tiempo'].dt.isocalendar().week.astype(int)

    # 2. División Train/Test para Series de Tiempo
    fecha_corte_test = pandas.to_datetime("2025-03-31") # Fecha fija para test

    train = data[data['indice_tiempo'] < fecha_corte_test]
    test = data[data['indice_tiempo'] == fecha_corte_test]

    # Definir características (X) y objetivo (y)
    features = ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year'] # Nuevas features
    target = 'pasajeros'

    X_train = train[features]
    X_train = X_train.astype('float64')  # Para no obtener el warning de NaN para enteros
    y_train = train[target]
    X_test = test[features]
    X_test = X_test.astype('float64')
    y_test = test[target]

    # 3. Modelo: Usaremos un Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) # n_jobs=-1 usa todos los cores
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. Log de Parámetros y Métricas
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("fecha_objetivo", fecha_corte_test.date().isoformat())
    mlflow.log_param("features_used", str(features))
    mlflow.log_metric("mae", mae)

    # Calculamos R2
    if len(y_test) > 1:
        score = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", score)
        print(f"R2: {score:.2f}")

    # Tags
    mlflow.set_tags({
        "data_split_strategy": "time_series_split",
        "model_complexity": "medium",
        "target_variable": "pasajeros"
    })

    # Log del modelo
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_rf_dia",
        input_example=input_example,
        signature=signature
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/modelo_rf_dia"
    result = mlflow.register_model(model_uri, "modelo_rf_dia")

    # Impresión de resultados
    print(f"Fecha real: {test['indice_tiempo'].iloc[0].date()}")
    print(f"Pasajeros Reales: {y_test.iloc[0]:,.0f}")
    print(f"Pasajeros Predichos: {y_pred[0]:,.0f}")
    print(f"MAE: {mae:,.0f}")
    # Fecha real: 2025-03-31
    # Pasajeros reales: 44,042
    # Pasajeros predichos: 44,404
    # MAE: 362

# Experimento 3: Predicción de Pasajeros Internacionales con Features de Tiempo y Evaluación en el Futuro (3 meses)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pasajeros Internacionales - Series de Tiempo (3 meses)")

# %%
with mlflow.start_run():
    data = dataset.copy()
    data = data[data['clasificacion_vuelo'] == 'Internacional']

    # 1. Feature Engineering
    # Extraemos información de la columna 'indice_tiempo'
    data['year'] = data['indice_tiempo'].dt.year
    data['month'] = data['indice_tiempo'].dt.month
    data['day_of_week'] = data['indice_tiempo'].dt.dayofweek # Lunes=0, Domingo=6
    data['day_of_year'] = data['indice_tiempo'].dt.dayofyear
    data['week_of_year'] = data['indice_tiempo'].dt.isocalendar().week.astype(int)

    # 2. División Train/Test para Series de Tiempo
    fecha_corte_test = data['indice_tiempo'].max() - pandas.DateOffset(months=3) # Últimos 3 meses para test

    train = data[data['indice_tiempo'] < fecha_corte_test]
    test = data[data['indice_tiempo'] >= fecha_corte_test]

    # Definir características (X) y objetivo (y)
    features = ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year']
    target = 'pasajeros'

    X_train = train[features]
    X_train = X_train.astype('float64')
    y_train = train[target]
    X_test = test[features]
    X_test = X_test.astype('float64')
    y_test = test[target]

    # 3. Modelo: Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 4. Log de Parámetros y Métricas
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)

    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))
    mlflow.log_param("fecha_objetivo", fecha_corte_test.date().isoformat())
    mlflow.log_param("features_used", str(features))
    mlflow.log_metric("mae", mae)

    # Calculamos R2
    if len(y_test) > 1:
        score = model.score(X_test, y_test)
        mlflow.log_metric("r2_score", score)
        print(f"R2: {score:.2f}")

    # Tags
    mlflow.set_tags({
        "data_split_strategy": "time_series_split",
        "model_complexity": "medium",
        "target_variable": "pasajeros"
    })

    # Log del modelo
    signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_rf_3meses",
        input_example=input_example,
        signature=signature
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/modelo_rf_3meses"
    result = mlflow.register_model(model_uri, "modelo_rf_3meses")

    # Impresión de resultados
    print(f"Fecha real: {test['indice_tiempo'].iloc[0].date()}")
    print(f"Pasajeros Reales: {y_test.iloc[0]:,.0f}")
    print(f"Pasajeros Predichos: {y_pred[0]:,.0f}")
    print(f"MAE: {mae:,.0f}")
    # Fecha real: 2024-12-31
    # Pasajeros reales: 36,751
    # Pasajeros predichos: 42,875
    # MAE: 9,602


# Experimento 4: Predicción de Pasajeros Internacionales con Variables Rezagadas y Evaluación en el Futuro (3 meses)

# En este experimento se predice la cantidad diaria de pasajeros en vuelos internacionales
# Se utilizan variables rezagadas (lags): pasajeros del día anterior, de hace una semana y la media móvil de los últimos 7 días.
# Se evalua el modelo sobre los últimos 3 meses del dataset.

import pandas as pd


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pasajeros Internacionales - Variables Rezagadas")

with mlflow.start_run():
    # 1. Filtrado y orden temporal
    data = dataset.copy()
    data = data[data['clasificacion_vuelo'] == 'Internacional'].reset_index(drop=True)
    
    # 2. Features de calendario
    data['day_of_week'] = data['indice_tiempo'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # 3. Variables rezagadas
    data['pasajeros_lag_1'] = data['pasajeros'].shift(1)
    data['pasajeros_lag_7'] = data['pasajeros'].shift(7)
    data['media_movil_7'] = data['pasajeros'].rolling(window=7).mean()

    # 4. Limpieza: eliminamos los primeros días sin datos válidos por el shift/rolling
    data = data.dropna().reset_index(drop=True)

    # 5. División Train/Test temporal (últimos 3 meses)
    fecha_corte_test = data['indice_tiempo'].max() - pd.DateOffset(months=3)
    train = data[data['indice_tiempo'] < fecha_corte_test]
    test = data[data['indice_tiempo'] >= fecha_corte_test]

    # 6. Definición de features y target
    features = ['day_of_week', 'is_weekend','asientos', 'vuelos','pasajeros_lag_1', 'pasajeros_lag_7', 'media_movil_7']
    target = 'pasajeros'

    X_train = train[features].astype('float64')
    y_train = train[target]
    X_test = test[features].astype('float64')
    y_test = test[target]

    # 7. Entrenamiento y predicción
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 8. Métricas
    mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    r2 = sklearn.metrics.r2_score(y_test, y_pred)

    # 9. Logging con MLflow
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": model.n_estimators,
        "features_used": str(features),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "fecha_inicio_train": train['indice_tiempo'].min().date().isoformat(),
        "fecha_fin_test": test['indice_tiempo'].max().date().isoformat()
    })
    mlflow.log_metrics({
        "mae": mae,
        "r2_score": r2
    })
    mlflow.set_tags({
        "feature_engineering": "lag_variables",
        "target_variable": "pasajeros",
        "data_split": "últimos_3_meses"
    })

    # 10. Registro del modelo
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_rf_lags",
        input_example=input_example,
        signature=signature
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/modelo_rf_lags"
    mlflow.register_model(model_uri, "modelo_rf_lags")

    # 11. Resultados impresos
    print(f"Fecha inicio test: {test['indice_tiempo'].min().date()}")
    print(f"Fecha fin test: {test['indice_tiempo'].max().date()}")
    print(f"MAE: {mae:,.0f}")
    print(f"R2: {r2:.2f}")
    print(f"Primer día real: {y_test.iloc[0]:,.0f}")
    print(f"Primer día predicho: {y_pred[0]:,.0f}")


# Experimento 5: Predicción de Pasajeros Internacionales con Lags de Pasajeros + Asientos/Vuelos/Lags
#
# Este experimento combina las variables rezagadas de pasajeros (como en el experimento 4)
# con variables operativas (`asientos`, `vuelos`), su derivada `asientos_promedio_por_vuelo`
# y sus respectivos lags. Se evalúa en los últimos 3 meses del dataset.


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Pasajeros Internacionales - Lags + Asientos/Vuelos")

with mlflow.start_run():
    data = dataset.copy()
    data = data[data['clasificacion_vuelo'] == 'Internacional'].reset_index(drop=True)

    # Filtrado para evitar divisiones por cero
    data = data[data['vuelos'] != 0]

    # Variables derivadas
    data['asientos_promedio_por_vuelo'] = data['asientos'] / data['vuelos']

    # Features de calendario
    data['day_of_week'] = data['indice_tiempo'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

    # Lags de pasajeros
    data['pasajeros_lag_1'] = data['pasajeros'].shift(1)
    data['pasajeros_lag_7'] = data['pasajeros'].shift(7)
    data['media_movil_7'] = data['pasajeros'].rolling(7).mean()

    # Lags de variables operativas
    data['asientos_lag_1'] = data['asientos'].shift(1)
    data['vuelos_lag_1'] = data['vuelos'].shift(1)
    data['asientos_promedio_lag_1'] = data['asientos_promedio_por_vuelo'].shift(1)

    # Limpieza
    data = data.dropna().reset_index(drop=True)

    # División temporal
    fecha_corte_test = data['indice_tiempo'].max() - pd.DateOffset(months=3)
    train = data[data['indice_tiempo'] < fecha_corte_test]
    test = data[data['indice_tiempo'] >= fecha_corte_test]

    # Features usadas
    features = [
        'day_of_week', 'is_weekend',
        'pasajeros_lag_1', 'pasajeros_lag_7', 'media_movil_7',
        'asientos', 'vuelos', 'asientos_promedio_por_vuelo',
        'asientos_lag_1', 'vuelos_lag_1', 'asientos_promedio_lag_1'
    ]
    target = 'pasajeros'

    X_train = train[features].astype('float64')
    y_train = train[target]
    X_test = test[features].astype('float64')
    y_test = test[target]

    # Modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # MLflow
    mlflow.log_params({
        "model_type": "RandomForestRegressor",
        "n_estimators": model.n_estimators,
        "features_used": str(features),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "fecha_inicio_train": train['indice_tiempo'].min().date().isoformat(),
        "fecha_fin_test": test['indice_tiempo'].max().date().isoformat()
    })
    mlflow.log_metrics({
        "mae": mae,
        "r2_score": r2
    })
    mlflow.set_tags({
        "feature_engineering": "pasajeros_lags + asientos/vuelos",
        "target_variable": "pasajeros",
        "data_split": "últimos_3_meses"
    })

    # Registro del modelo
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, model.predict(X_train))
    input_example = X_train.iloc[:5]
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="modelo_rf_lags_operativos",
        input_example=input_example,
        signature=signature
    )
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/modelo_rf_lags_operativos"
    mlflow.register_model(model_uri, "modelo_rf_lags_operativos")

    # Resultados
    print(f"Fecha inicio test: {test['indice_tiempo'].min().date()}")
    print(f"Fecha fin test: {test['indice_tiempo'].max().date()}")
    print(f"MAE: {mae:,.0f}")
    print(f"R2: {r2:.2f}")
    print(f"Primer día real: {y_test.iloc[0]:,.0f}")
    print(f"Primer día predicho: {y_pred[0]:,.0f}")
