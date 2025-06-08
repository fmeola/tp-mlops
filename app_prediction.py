# %%
# App que levanta mlflow y el deploy de fastapi

import os
import subprocess
import threading

import fastapi
import mlflow
import pandas
import uvicorn

# Configuración MLFlow
URI = 'http://127.0.0.1:5000'

# Predicción A: Pasajeros internacionales del día 31/03/2025
INTERNATIONAL_PASSENGERS_MODEL = 'pasajeros_internacionales_dia'

def obtener_mejor_modelo(model_name):
    mlflow.set_tracking_uri(URI)
    client = mlflow.tracking.MlflowClient()
    # Obtener todas las versiones del modelo
    versions = client.get_latest_versions(name=model_name, stages=["None", "Staging", "Production"])
    best_run = None
    best_mae = float("inf")
    for v in versions:
        run = client.get_run(v.run_id)
        if "mae" in run.data.metrics:
            mae = run.data.metrics["mae"]
            if mae < best_mae:
                best_mae = mae
                best_run = run
    if not best_run:
        raise Exception("No hay versiones del modelo con la métrica 'mae' registrada")
    model_uri = f"runs:/{best_run.info.run_id}/{model_name}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, best_run

def construir_features(fecha_objetivo, run, features):
    # Fecha de inicio registrada en el experimento
    fecha_inicio = pandas.to_datetime(run.data.params.get("fecha_inicio", "2017-01-01"))
    # Variables temporales calculadas desde la fecha
    datos = {
        # Experimento 1
        "dias_desde_inicio": (fecha_objetivo - fecha_inicio).days,
        # Experimento 2
        "year": fecha_objetivo.year,
        "month": fecha_objetivo.month,
        "day_of_week": fecha_objetivo.dayofweek,
        "day_of_year": fecha_objetivo.dayofyear,
        "week_of_year": fecha_objetivo.isocalendar().week
    }
    # Filtrar solo las que requiere el modelo
    datos_filtrados = {k: datos[k] for k in features if k in datos}
    # Devolver DataFrame con una sola fila
    return pandas.DataFrame([datos_filtrados]).astype("float64")

app = fastapi.FastAPI(title = 'Pasajeros Internacionales')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/pasajeros_internacionales")
def predict(fecha: str = "2025-03-31"):
    try:
        # Parsear la fecha objetivo
        fecha_objetivo = pandas.to_datetime(fecha)
        # Obtener el mejor modelo y corrida
        model, run = obtener_mejor_modelo(INTERNATIONAL_PASSENGERS_MODEL)
        # Obtener el nombre de las features usadas
        features_used = eval(run.data.params.get("features_used"))
        # Construir DataFrame de entrada según las features del modelo
        df = construir_features(fecha_objetivo, run, features_used)
        # Predicción con el mejor modelo
        prediction = model.predict(df).tolist()
        # Respuesta
        return {
            "modelo_usado": run.info.run_id,
            "features": features_used,
            "fecha": fecha_objetivo.date().isoformat(),
            "pasajeros_predichos": f"{prediction[0]:,.0f}"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=fastapi.responses.FileResponse)
def serve_index():
    return fastapi.responses.FileResponse(os.path.join("templates", "index.html"))

if __name__ == "__main__":
    def start_mlflow():
        subprocess.run(["mlflow", "ui", "--port", "5000", "--host", "127.0.0.1"])

    hilo = threading.Thread(target=start_mlflow, daemon=True)
    hilo.start()

    uvicorn.run("app_prediction:app", host="127.0.0.1", port=8000, reload=True)