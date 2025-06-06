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
