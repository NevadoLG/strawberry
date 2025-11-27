// api/static/js/batch.js
const batchForm = document.getElementById("batch-form");
const batchFilesInput = document.getElementById("batch_files");
const resultCard = document.getElementById("batch-result-card");
const batchMetaDiv = document.getElementById("batch-meta");
const batchTableBody = document.querySelector("#batch-table tbody");
const saveBatchHistoryBtn = document.getElementById("save-batch-history");

let lastBatchResponse = null; // guardamos el último resultado del lote

batchForm.addEventListener("submit", async (e) => {
  e.preventDefault();

  if (!batchFilesInput.files.length) {
    alert("Selecciona al menos una imagen para el lote.");
    return;
  }

  const loteId = document.getElementById("lote_id").value.trim();
  const location = document.getElementById("location").value.trim();
  const description = document.getElementById("description").value.trim();

  if (!loteId) {
    alert("El ID de lote es obligatorio.");
    return;
  }

  const formData = new FormData();
  formData.append("lote_id", loteId);
  formData.append("location", location);
  formData.append("description", description);

  for (const file of batchFilesInput.files) {
    formData.append("files", file);
  }

  // limpiar resultados previos
  batchTableBody.innerHTML = "";
  batchMetaDiv.innerHTML = "";
  resultCard.style.display = "none";
  lastBatchResponse = null;
  saveBatchHistoryBtn.disabled = true;

  try {
    const response = await fetch("/predict-batch", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || `Error HTTP ${response.status}`);
    }

    const data = await response.json();
    lastBatchResponse = data;

    batchMetaDiv.innerHTML = `
      <p><strong>ID de lote:</strong> ${data.batch_meta.lote_id}</p>
      <p><strong>Ubicación:</strong> ${
        data.batch_meta.location || "(no registrada)"
      }</p>
      <p><strong>Descripción:</strong> ${
        data.batch_meta.description || "(sin descripción)"
      }</p>
      <p><strong>Número de archivos procesados:</strong> ${data.num_files}</p>
    `;
    // guardamos meta básica en el objeto por si acaso
    lastBatchResponse.ui_meta = {
      lote_id: data.batch_meta.lote_id,
      location: data.batch_meta.location,
      description: data.batch_meta.description,
      num_files: data.num_files,
    };

    data.results.forEach((item) => {
      const tr = document.createElement("tr");

      const tdFile = document.createElement("td");
      tdFile.textContent = item.filename || "-";

      const tdNumDet = document.createElement("td");
      tdNumDet.textContent =
        item.num_detections !== undefined ? item.num_detections : "-";

      const tdClasses = document.createElement("td");
      if (item.detections && Array.isArray(item.detections)) {
        const classNames = [
          ...new Set(item.detections.map((d) => d.class_name)),
        ];
        tdClasses.textContent = classNames.join(", ") || "-";
      } else {
        tdClasses.textContent = "-";
      }

      const tdConf = document.createElement("td");
      if (item.max_confidence !== undefined) {
        tdConf.textContent = item.max_confidence.toFixed(2);
      } else {
        tdConf.textContent = "-";
      }

      const tdStatus = document.createElement("td");
      tdStatus.textContent = item.error ? "Error" : "OK";

      tr.appendChild(tdFile);
      tr.appendChild(tdNumDet);
      tr.appendChild(tdClasses);
      tr.appendChild(tdConf);
      tr.appendChild(tdStatus);

      batchTableBody.appendChild(tr);
    });

    resultCard.style.display = "block";
    saveBatchHistoryBtn.disabled = false;
  } catch (err) {
    console.error(err);
    alert(`Ocurrió un error al procesar el lote: ${err.message}`);
  }
});

// --------- Guardar lote en historial (localStorage) ---------
saveBatchHistoryBtn.addEventListener("click", () => {
  if (!lastBatchResponse) {
    alert("Primero procesa un lote antes de guardarlo en el historial.");
    return;
  }

  const raw = localStorage.getItem("sv_history");
  let history = [];
  try {
    if (raw) {
      history = JSON.parse(raw);
      if (!Array.isArray(history)) history = [];
    }
  } catch {
    history = [];
  }

  const now = new Date();
  const meta = lastBatchResponse.ui_meta || lastBatchResponse.batch_meta || {};

  const entry = {
    id: `batch-${now.getTime()}`,
    type: "batch",
    created_at: now.toISOString(),
    meta: {
      lote_id: meta.lote_id || "",
      location: meta.location || "",
      description: meta.description || "",
      num_files: meta.num_files || lastBatchResponse.num_files || 0,
    },
    payload: lastBatchResponse,
  };

  history.push(entry);
  localStorage.setItem("sv_history", JSON.stringify(history));

  alert("Lote guardado en el historial local de este navegador.");
});
