const selectModelo = document.getElementById("selectModelo");
const atributosContainer = document.getElementById("atributosContainer");
const btnClasificar = document.getElementById("btnClasificar");

// Guardamos aqui el dataset de origen del modelo elegido, para mandarlo
// como contexto al LLM al momento de pedir la explicacion.
let datasetDeOrigen = "";

async function cargarModelos() {
  const res = await fetch("/api/models");
  const modelos = await res.json();
  selectModelo.innerHTML = modelos.map(m => `<option value="${m.replace('.joblib','')}">${m}</option>`).join("");
  if (modelos.length) cargarAtributosDeModelo(selectModelo.value);
}

// Genera dinamicamente Atrib1, Atrib2 ... AtribN leyendo los atributos
// que quedaron guardados dentro del modelo entrenado.
async function cargarAtributosDeModelo(modelName) {
  if (!modelName) return;
  const res = await fetch(`/api/atributos_modelo?model_name=${modelName}`);
  const info = await res.json();

  if (info.error) {
    atributosContainer.innerHTML = `<p>${info.error}</p>`;
    return;
  }

  datasetDeOrigen = info.dataset_filename;
  atributosContainer.innerHTML = info.atributos.map(attr => `
    <div class="atributo-row">
      <label>${attr}</label>
      <input type="number" step="any" data-attr="${attr}">
    </div>
  `).join("");
}

selectModelo.addEventListener("change", (e) => cargarAtributosDeModelo(e.target.value));

btnClasificar.addEventListener("click", async () => {
  const atributos = {};
  document.querySelectorAll("[data-attr]").forEach(input => {
    atributos[input.dataset.attr] = input.value;
  });

  const res = await fetch("/api/clasificar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: selectModelo.value,
      atributos,
      contexto_dataset: datasetDeOrigen,
    }),
  });
  const data = await res.json();

  document.getElementById("resultadoTexto").textContent = data.resultado ?? data.error;
  document.getElementById("explicacionTexto").textContent = data.explicacion ?? "";
});

cargarModelos();
