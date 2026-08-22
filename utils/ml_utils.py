"""
utils/ml_utils.py
Funciones para: cargar datasets, entrenar los 3 algoritmos de ML,
calcular metricas (precision, recall, f1, support) y clasificar
un ejemplo nuevo con un modelo ya guardado.
"""
import os
import base64
from io import BytesIO

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # sin GUI, para poder correr dentro de Flask
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# --- Los scripts de algoritmos de ML del boceto: ID3 y K-NN,
#     con los mismos hiperparametros que ya tenias en tus .py ---
ALGORITHMS = {
    "id3": lambda: DecisionTreeClassifier(criterion="entropy", random_state=42),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
}


def list_datasets():
    return [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]


def list_models():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]


def load_dataset(filename):
    path = os.path.join(DATASETS_DIR, filename)
    return pd.read_csv(path)


def get_attributes(filename, target_column):
    """Regresa la lista de atributos (columnas) excluyendo la columna objetivo.
    Esto es lo que llena dinamicamente Atrib1, Atrib2 ... AtribN en la pantalla Clasificar."""
    df = load_dataset(filename)
    return [c for c in df.columns if c != target_column]


def get_model_attributes(model_name):
    """Lee un modelo ya guardado y regresa sus atributos esperados (en el
    orden correcto de entrenamiento) y el dataset del que salio, sin
    necesidad de que el usuario vuelva a elegir el dataset en la pantalla
    Clasificar."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    bundle = joblib.load(model_path)
    return {
        "atributos": bundle["feature_columns"],
        "dataset_filename": bundle["dataset_filename"],
    }


def train_model(algoritmo, dataset_filename, target_column, holdout_pct, model_name):
    """
    algoritmo: 'random_forest' | 'svm' | 'logistic_regression'
    holdout_pct: porcentaje (0-100) que se separa como set de prueba
    model_name: nombre con el que se guarda el .joblib (debe ser el mismo
                nombre que luego se selecciona en la pantalla Clasificar)
    """
    if algoritmo not in ALGORITHMS:
        raise ValueError(f"Algoritmo no soportado: {algoritmo}")

    df = load_dataset(dataset_filename)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Si la clase es texto (ej. 'versicolor'), la codificamos para el modelo
    # pero guardamos el encoder junto con el modelo para poder revertirlo despues.
    label_encoder = None
    if y.dtype == object:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    test_size = holdout_pct / 100.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    clf = ALGORITHMS[algoritmo]()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # target_names para que el reporte muestre "versicolor" en vez de 0,1,2
    target_names = None
    if label_encoder is not None:
        target_names = [str(c) for c in label_encoder.classes_]

    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0
    )

    # Matriz de confusion -> imagen base64 para mostrar en la pantalla Entrenar
    # (esto es lo que resuelve el "True Positive / False Positive?" del pizarron:
    # en un problema de N clases, TP/FP/FN/TN se leen por clase a partir de esta matriz)
    cm = confusion_matrix(y_test, y_pred)
    labels = target_names if target_names else sorted(set(y_test) | set(y_pred))
    confusion_matrix_b64 = _plot_confusion_matrix(cm, labels, algoritmo)

    # Guardamos modelo + encoder + columnas + dataset origen en un solo joblib
    bundle = {
        "model": clf,
        "label_encoder": label_encoder,
        "feature_columns": list(X.columns),
        "target_column": target_column,
        "dataset_filename": dataset_filename,
        "algoritmo": algoritmo,
    }
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(bundle, model_path)

    return report_dict, confusion_matrix_b64


def _plot_confusion_matrix(cm, labels, algoritmo):
    """Genera la matriz de confusion con seaborn y la regresa como
    string base64 (data:image/png) para incrustarla directo en el <img> del HTML."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Matriz de Confusion ({algoritmo})")
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def classify_example(model_name, atributos: dict):
    """
    atributos: dict {nombre_columna: valor} capturado desde los inputs
               dinamicos Atrib1..AtribN de la pantalla Clasificar.
    Regresa: (clase_predicha, dict_de_atributos_usado)
    """
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    bundle = joblib.load(model_path)

    clf = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    # Ordenamos los valores igual que las columnas de entrenamiento
    fila = [[float(atributos[col]) for col in feature_columns]]
    pred = clf.predict(fila)[0]

    if label_encoder is not None:
        pred = label_encoder.inverse_transform([pred])[0]

    return pred, atributos
