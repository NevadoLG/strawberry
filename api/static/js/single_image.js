// api/static/js/single_image.js
document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("file-input");
  const selectBtn = document.getElementById("select-btn");
  const analyzeBtn = document.getElementById("analyze-btn");
  const previewImg = document.getElementById("preview-image");
  const statusEl = document.getElementById("status");
  const resultImg = document.getElementById("annotated-image");
  const resultWrapper = document.getElementById("result-annotated-wrapper");
  const humanSummary = document.getElementById("human-summary");
  const jsonOutput = document.getElementById("json-output");
  const saveHistoryBtn = document.getElementById("save-history-single");

  let currentFile = null;
  let lastResultData = null; // <-- aquí guardamos el último resultado

  function setFile(file) {
    if (!file) return;
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.classList.remove("hidden");
    };
    reader.readAsDataURL(file);

    analyzeBtn.disabled = false;
    statusEl.textContent = "";
  }

  function buildHumanSummary(data) {
    if (!data || data.num_detections === 0) {
      humanSummary.innerHTML = `
        <p>No se detectaron fresas en la imagen.</p>
      `;
      return;
    }

    const summary = data.summary_by_class || {};
    const total = data.num_detections || 0;

    let listItems = "";
    for (const [cls, info] of Object.entries(summary)) {
      const count = info.count ?? 0;
      const best = (info.best_score ?? 0).toFixed(2);
      listItems += `
        <li>
          <strong>${cls}</strong>: ${count} detección(es)
          <span class="sv-chip-score">mejor score: ${best}</span>
        </li>
      `;
    }

    humanSummary.innerHTML = `
      <p><strong>Total de detecciones:</strong> ${total}</p>
      <ul class="sv-summary-list">
        ${listItems}
      </ul>
    `;
  }

  // --------- drag & drop ---------
  ["dragenter", "dragover"].forEach((eventName) => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropArea.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropArea.classList.remove("dragover");
    });
  });

  dropArea.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (!dt || !dt.files || !dt.files.length) return;
    const file = dt.files[0];
    setFile(file);

    // sincronizamos con el input para el FormData
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
  });

  selectBtn.addEventListener("click", () => {
    fileInput.click();
  });

  dropArea.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      setFile(fileInput.files[0]);
    }
  });

  analyzeBtn.addEventListener("click", async () => {
    if (!fileInput.files || !fileInput.files[0]) {
      alert("Selecciona una imagen primero.");
      return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    statusEl.textContent = "Analizando imagen...";
    analyzeBtn.disabled = true;
    saveHistoryBtn.disabled = true;
    lastResultData = null;

    try {
      const resp = await fetch("/predict?conf=0.4", {
        method: "POST",
        body: formData,
      });

      const data = await resp.json();

      if (!resp.ok) {
        statusEl.textContent = "Error: " + (data.detail || "desconocido");
        analyzeBtn.disabled = false;
        return;
      }

      statusEl.textContent = "OK";

      if (data.annotated_image_base64) {
        resultImg.src = "data:image/png;base64," + data.annotated_image_base64;
        resultWrapper.classList.remove("hidden");
      }

      buildHumanSummary(data);

      const debugCopy = Object.assign({}, data);
      delete debugCopy.annotated_image_base64;
      jsonOutput.textContent = JSON.stringify(debugCopy, null, 2);

      // Guardamos el último resultado en memoria para el botón de guardar
      lastResultData = data;
      saveHistoryBtn.disabled = false;
    } catch (err) {
      console.error(err);
      statusEl.textContent = "Error de red o del servidor.";
    } finally {
      analyzeBtn.disabled = false;
    }
  });

  // --------- Guardar en historial (localStorage) ---------
  saveHistoryBtn.addEventListener("click", () => {
    if (!lastResultData) {
      alert("Primero analiza una imagen antes de guardar en el historial.");
      return;
    }

    const historyRaw = localStorage.getItem("sv_history");
    let history = [];
    try {
      if (historyRaw) {
        history = JSON.parse(historyRaw);
        if (!Array.isArray(history)) history = [];
      }
    } catch {
      history = [];
    }

    const now = new Date();
    const entry = {
      id: `single-${now.getTime()}`,
      type: "single",
      created_at: now.toISOString(),
      meta: {
        filename: currentFile ? currentFile.name : "imagen_sin_nombre",
        num_detections: lastResultData.num_detections || 0,
      },
      payload: lastResultData,
    };

    history.push(entry);
    localStorage.setItem("sv_history", JSON.stringify(history));

    alert("Resultado guardado en el historial local de este navegador.");
  });
});
