# Trabajo Práctico Final - MLOps

## Eje Temático
- Experimentación y comparación de modelos con MLflow

## Integrantes
- Castillo Maira
- Fera Camila
- Izzo Paula Ruth
- Meola Franco Román

## Instrucciones

1. Ejecutar `app_prediction.py` para iniciar mlflow y fastapi

```bash
python app_prediction.py
```

2. Ejecutar `run_experiment.py` para cargar un modelo y hacer el deploy en fastapi

```bash
python run_experiment.py
```

3. Consultar la predicción haciendo un GET HTTP al endpoint predict

```http request
GET 127.0.0.1:8000/predict
```
