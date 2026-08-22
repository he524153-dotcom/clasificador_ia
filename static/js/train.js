const selectDataset = document.getElementById("selectDataset");
const btnTrain = document.getElementById("btnTrain");
const tablaBody = document.querySelector("#tablaResultados tbody");

async function cargarDatasets() {
  const res = await fetch("/api/datasets");
  const datasets = await res.json();
  selectDataset.innerHTML = datasets.map(d => `<option value="${d}">${d}</option>`).join("");
}

btnTrain.addEventListener("click", async () => {
  const payload = {
    algoritmo: document.getElementById("selectAlgoritmo").value,
    dataset: selectDataset.value,
    target_column: document.getElementById("targetColumn").value,
    model_name: document.getElementById("modelName").value,
    holdout_pct: document.getElementById("holdoutPct").value,
  };

  const res = await fetch("/api/entrenar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();

  if (data.error) {
    alert(data.error);
    return;
  }

  // data.resultado es el output_dict de classification_report:
  // { "clase1": {precision, recall, f1-score, support}, ..., "accuracy": .., "macro avg": {...}, "weighted avg": {...} }
  tablaBody.innerHTML = "";
  for (const [clase, metrics] of Object.entries(data.resultado)) {
    if (clase === "accuracy") continue; // accuracy es un solo numero, no una fila con las 4 metricas
    tablaBody.innerHTML += `
      <tr>
        <td>${clase}</td>
        <td>${metrics.precision.toFixed(2)}</td>
        <td>${metrics.recall.toFixed(2)}</td>
        <td>${metrics["f1-score"].toFixed(2)}</td>
        <td>${metrics.support}</td>
      </tr>`;
  }

  // Muestra la matriz de confusion (resuelve el TP/FP/FN/TN del pizarron)
  const img = document.getElementById("confMatrixImg");
  if (data.matriz_confusion) {
    img.src = data.matriz_confusion;
    img.style.display = "block";
  }
});

cargarDatasets();
