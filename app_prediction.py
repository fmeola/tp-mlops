# %%
# App que levanta mlflow y el deploy de fastapi

import subprocess
import threading

import fastapi
import mlflow
import pandas
import uvicorn

def obtener_mejor_modelo():
    # Configuración MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()
    # Obtener todas las corridas del experimento
    experiment = client.get_experiment_by_name("Pasajeros Internacionales")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.mae ASC"],
        max_results=20
    )
    # Filtrar las corridas que tienen mae
    runs_with_mae = [run for run in runs if "mae" in run.data.metrics]
    # Tomar la mejor corrida (por menor MAE)
    best_run = sorted(runs_with_mae, key=lambda r: r.data.metrics["mae"])[0]
    model_uri = f"runs:/{best_run.info.run_id}/modelo_regresion_dia"
    model = mlflow.pyfunc.load_model(model_uri)
    return model, best_run

app = fastapi.FastAPI(title = 'Pasajeros Internacionales')

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict():
    try:
        # Obtener el mejor modelo y corrida
        model, run = obtener_mejor_modelo()
        # Obtener dias_desde_inicio_prediccion de la corrida
        dias_desde_inicio = int(run.data.params["dias_desde_inicio_prediccion"])
        # Predicción con el mejor modelo
        df = pandas.DataFrame([[dias_desde_inicio]], columns=["dias_desde_inicio"]).astype('float64')
        prediction = model.predict(df).tolist()
        # Respuesta
        return {
            "pasajeros_predichos": f"{prediction[0]:,.0f}"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    def start_mlflow():
        subprocess.run(["mlflow", "ui", "--port", "5000", "--host", "127.0.0.1"])

    hilo = threading.Thread(target=start_mlflow, daemon=True)
    hilo.start()

    uvicorn.run("app_prediction:app", host="127.0.0.1", port=8000, reload=True)