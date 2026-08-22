import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from utils import ml_utils, llm_utils

load_dotenv()  # lee GROQ_API_KEY del .env

BASE_DIR = os.path.dirname(__file__)
app = Flask(__name__)
app.config["DATASETS_DIR"] = os.path.join(BASE_DIR, "datasets")
app.config["MODELS_DIR"] = os.path.join(BASE_DIR, "models")


# ---------- Rutas de pantallas (las 2 vistas del boceto) ----------

@app.route("/")
def index():
    return render_template("clasificar.html")


@app.route("/clasificar")
def clasificar_view():
    return render_template("clasificar.html")


@app.route("/entrenar")
def entrenar_view():
    return render_template("entrenar.html")


# ---------- APIs comunes ----------

@app.route("/api/datasets", methods=["GET"])
def api_list_datasets():
    return jsonify(ml_utils.list_datasets())


@app.route("/api/models", methods=["GET"])
def api_list_models():
    return jsonify(ml_utils.list_models())


@app.route("/api/upload_dataset", methods=["POST"])
def api_upload_dataset():
    file = request.files.get("dataset")
    if not file or not file.filename.endswith(".csv"):
        return jsonify({"error": "Sube un archivo .csv valido"}), 400
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["DATASETS_DIR"], filename))
    return jsonify({"filename": filename})


@app.route("/api/atributos", methods=["GET"])
def api_atributos():
    """Regresa las columnas del dataset (sin la columna objetivo) para
    construir dinamicamente Atrib1, Atrib2 ... AtribN en el formulario."""
    dataset = request.args.get("dataset")
    target_column = request.args.get("target_column", "species")
    try:
        atributos = ml_utils.get_attributes(dataset, target_column)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(atributos)


@app.route("/api/atributos_modelo", methods=["GET"])
def api_atributos_modelo():
    """Igual que /api/atributos, pero lee los atributos directo del modelo
    ya entrenado (guardados dentro del .joblib), en vez de pedirle al
    usuario que vuelva a elegir el dataset en la pantalla Clasificar."""
    model_name = request.args.get("model_name")
    try:
        info = ml_utils.get_model_attributes(model_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(info)


# ---------- Pantalla Entrenar ----------

@app.route("/api/entrenar", methods=["POST"])
def api_entrenar():
    data = request.get_json()
    algoritmo = data.get("algoritmo")
    dataset = data.get("dataset")
    target_column = data.get("target_column", "species")
    holdout_pct = float(data.get("holdout_pct", 20))
    # el modelo se guarda con este nombre, y con ESE MISMO nombre se debe
    # seleccionar despues en la pantalla Clasificar
    model_name = data.get("model_name")

    try:
        reporte, confusion_matrix_b64 = ml_utils.train_model(
            algoritmo, dataset, target_column, holdout_pct, model_name
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"resultado": reporte, "matriz_confusion": confusion_matrix_b64})


# ---------- Pantalla Clasificar ----------

@app.route("/api/clasificar", methods=["POST"])
def api_clasificar():
    data = request.get_json()
    model_name = data.get("model_name")
    atributos = data.get("atributos", {})
    contexto = data.get("contexto_dataset", "")

    try:
        clase, atributos_usados = ml_utils.classify_example(model_name, atributos)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        explicacion = llm_utils.explicar_clasificacion(clase, atributos_usados, contexto)
    except Exception as e:
        explicacion = f"(No se pudo generar explicacion con el LLM: {e})"

    return jsonify({"resultado": clase, "explicacion": explicacion})


if __name__ == "__main__":
    os.makedirs(app.config["DATASETS_DIR"], exist_ok=True)
    os.makedirs(app.config["MODELS_DIR"], exist_ok=True)
    app.run(debug=True, port=5000)
