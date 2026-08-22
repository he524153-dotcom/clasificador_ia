# Sistema Experto - GUI Flask (Clasificar / Entrenar)

## Instalación
```bash
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # y pega tu GROQ_API_KEY real
python app.py
```
Abre http://localhost:5000

## Cómo mapea a tus bocetos

**Boceto "Clasificar"**: `templates/clasificar.html` + `static/js/classify.js`.
Al elegir un dataset, `GET /api/atributos` regresa las columnas (menos la
clase) y el JS genera los inputs `Atrib1..AtribN` dinámicamente. Al dar
"Classify", `POST /api/clasificar` predice con el modelo `.joblib` elegido
y llama a `utils/llm_utils.py` (Llama-3.3-70B vía Groq) para generar el
texto de "Explicación".

**Boceto "Entrenar"**: `templates/entrenar.html` + `static/js/train.js`.
`POST /api/entrenar` entrena uno de los 3 algoritmos (`utils/ml_utils.py`
-> `ALGORITHMS`), separa el hold-out con el % indicado, calcula
`classification_report` (precision/recall/f1-score/support) y guarda el
modelo con `joblib` en `models/<model_name>.joblib`.

**Nombre de modelo consistente**: el `model_name` que pones al entrenar
es el mismo nombre que después aparece en el `<select>` de la pantalla
Clasificar (viene de `GET /api/models`, que lista los `.joblib` guardados).

## Agregar tus propios datasets
Sube un `.csv` desde el botón "Upload" en la pantalla Clasificar, o
copia el archivo directamente a la carpeta `datasets/`. La última
columna (o la que indiques como `target_column`) debe ser la clase.

## Notas
- Para usar otro proveedor de Llama-3.3-70B (Together.ai, Fireworks, etc.)
  solo cambia el cliente dentro de `utils/llm_utils.py`; la firma de
  `explicar_clasificacion()` no cambia.
- `SVC` se instancia con `probability=True` por si luego quieres mostrar
  también la probabilidad de la clase predicha.
