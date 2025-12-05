// static/js/results.js

// --- Utilidades ---
function loadHistory() {
  const raw = localStorage.getItem("sv_history");
  console.log("[results] raw sv_history =", raw);

  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    console.error("[results] Error parseando sv_history:", e);
    return [];
  }
}

function formatDate(iso) {
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso || "";
  }
}

function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "block";
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
}

// --- Lógica principal ---
document.addEventListener("DOMContentLoaded", () => {
  console.log("[results] DOMContentLoaded");

  const history = loadHistory();
  console.log("[results] history:", history);

  const noHistoryMsg = document.getElementById("no-history-msg");
  const singleCard = document.getElementById("single-history-card");
  const batchCard = document.getElementById("batch-history-card");
  const clearBtn = document.getElementById("clear-history");

  if (!history.length) {
    noHistoryMsg.style.display = "block";
    singleCard.style.display = "none";
    batchCard.style.display = "none";
    return;
  }

  noHistoryMsg.style.display = "none";

  const singles = history.filter((h) => h.type === "single");
  const batches = history.filter((h) => h.type === "batch");

  // ========= CLASIFICACIÓN ÚNICA =========
  const singleTbody = document.querySelector("#single-history-table tbody");
  if (singles.length) {
    singleCard.style.display = "block";
    singleTbody.innerHTML = "";

    singles.forEach((item) => {
      const tr = document.createElement("tr");

      const tdFecha = document.createElement("td");
      tdFecha.textContent = formatDate(item.created_at);

      const tdArchivo = document.createElement("td");
      tdArchivo.textContent = item.meta?.filename || "(sin nombre)";

      const tdNumDet = document.createElement("td");
      tdNumDet.textContent = item.meta?.num_detections ?? "-";

      const tdVer = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = "sv-btn sv-btn-secondary sv-btn-xs";
      btn.textContent = "Ver";
      btn.addEventListener("click", () => showSingleModal(item));
      tdVer.appendChild(btn);

      tr.appendChild(tdFecha);
      tr.appendChild(tdArchivo);
      tr.appendChild(tdNumDet);
      tr.appendChild(tdVer);

      singleTbody.appendChild(tr);
    });
  } else {
    singleCard.style.display = "none";
  }

  // ========= LOTES =========
  const batchTbody = document.querySelector("#batch-history-table tbody");
  if (batches.length) {
    batchCard.style.display = "block";
    batchTbody.innerHTML = "";

    batches.forEach((entry) => {
      const meta = entry.meta || {};
      const tr = document.createElement("tr");

      const tdFecha = document.createElement("td");
      tdFecha.textContent = formatDate(entry.created_at);

      const tdLote = document.createElement("td");
      tdLote.textContent = meta.lote_id || "(sin ID)";

      const tdLoc = document.createElement("td");
      tdLoc.textContent = meta.location || "(sin ubicación)";

      const tdNumFiles = document.createElement("td");
      tdNumFiles.textContent = meta.num_files ?? "-";

      const tdVer = document.createElement("td");
      const btn = document.createElement("button");
      btn.className = "sv-btn sv-btn-secondary sv-btn-xs";
      btn.textContent = "Ver resumen";
      btn.addEventListener("click", () => showBatchModal(entry));
      tdVer.appendChild(btn);

      tr.appendChild(tdFecha);
      tr.appendChild(tdLote);
      tr.appendChild(tdLoc);
      tr.appendChild(tdNumFiles);
      tr.appendChild(tdVer);

      batchTbody.appendChild(tr);
    });
  } else {
    batchCard.style.display = "none";
  }

  // ========= LIMPIAR HISTORIAL =========
  clearBtn.addEventListener("click", () => {
    if (!confirm("¿Seguro que deseas borrar todo el historial local?")) return;
    localStorage.removeItem("sv_history");
    location.reload();
  });

  // ========= CIERRE DE MODALES =========
  document.querySelectorAll(".sv-modal-close").forEach((btn) => {
    const target = btn.getAttribute("data-modal-close");
    btn.addEventListener("click", () => closeModal(target));
  });

  document.querySelectorAll(".sv-modal-backdrop").forEach((bg) => {
    const parent = bg.closest(".sv-modal");
    if (!parent) return;
    bg.addEventListener("click", () => (parent.style.display = "none"));
  });
});

// ===================
// MODAL IMAGEN ÚNICA
// ===================
function showSingleModal(entry) {
  const payload = entry.payload || {};
  const detections = payload.detections || [];
  const filename = entry.meta?.filename || "(sin nombre)";

  const imgBase64 = payload.annotated_image_base64;
  const imgEl = document.getElementById("single-modal-img");
  const titleEl = document.getElementById("single-modal-title");
  const infoEl = document.getElementById("single-modal-info");
  const listEl = document.getElementById("single-modal-list");

  titleEl.textContent = filename;
  infoEl.textContent = `Detecciones: ${detections.length || 0}`;
  listEl.innerHTML = "";

  detections.forEach((d, i) => {
  const li = document.createElement("li");

  let confValue = null;
  if (typeof d.confidence === "number") {
    confValue = d.confidence;
  } else if (typeof d.score === "number") {
    confValue = d.score;
  }

  const confText = confValue !== null ? confValue.toFixed(2) : "?";

  li.textContent = `${i + 1}. ${d.class_name} (confianza: ${confText})`;
  listEl.appendChild(li);
});


  if (imgBase64) {
    imgEl.src = `data:image/png;base64,${imgBase64}`;
  } else {
    imgEl.src = "";
  }

  openModal("single-modal");
}

// ===================
// MODAL LOTE
// ===================
function showBatchModal(entry) {
  const payload = entry.payload || {};
  const meta = entry.meta || {};
  const results = payload.results || [];

  const titleEl = document.getElementById("batch-modal-title");
  const infoEl = document.getElementById("batch-modal-info");
  const chartEl = document.getElementById("batch-classes-chart");
  const tbody = document.querySelector("#batch-modal-table tbody");

  titleEl.textContent = meta.lote_id || "Lote sin ID";
  infoEl.textContent = `Ubicación: ${
    meta.location || "sin ubicación"
  } · Archivos: ${meta.num_files || results.length}`;

  // tabla detalle
  tbody.innerHTML = "";
  results.forEach((r) => {
    const tr = document.createElement("tr");

    const tdFile = document.createElement("td");
    tdFile.textContent = r.filename || "-";

    const tdNum = document.createElement("td");
    tdNum.textContent = r.num_detections ?? "-";

    const tdClasses = document.createElement("td");
    tdClasses.textContent = (r.classes || []).join(", ") || "-";

    const tdConf = document.createElement("td");
    tdConf.textContent =
      r.max_confidence !== null && r.max_confidence !== undefined
        ? r.max_confidence.toFixed(2)
        : "-";

    const tdStatus = document.createElement("td");
    tdStatus.textContent = r.status || "-";

    tr.appendChild(tdFile);
    tr.appendChild(tdNum);
    tr.appendChild(tdClasses);
    tr.appendChild(tdConf);
    tr.appendChild(tdStatus);

    tbody.appendChild(tr);
  });

  // gráfico sencillo de clases
  const counts = {};
  results.forEach((r) => {
    (r.classes || []).forEach((cls) => {
      counts[cls] = (counts[cls] || 0) + 1;
    });
  });

  chartEl.innerHTML = "";
  const entries = Object.entries(counts);
  if (!entries.length) {
    chartEl.textContent = "No hay clases registradas en este lote.";
  } else {
    const max = Math.max(...entries.map(([, v]) => v));
    entries
      .sort((a, b) => b[1] - a[1])
      .forEach(([cls, count]) => {
        const row = document.createElement("div");
        row.className = "batch-classes-chart-row";

        const label = document.createElement("div");
        label.className = "batch-classes-chart-label";
        label.textContent = cls;

        const barWrapper = document.createElement("div");
        barWrapper.className = "batch-classes-chart-bar-wrapper";

        const bar = document.createElement("div");
        bar.className = "batch-classes-chart-bar";
        bar.style.width = `${(count / max) * 100}%`;

        barWrapper.appendChild(bar);

        const value = document.createElement("div");
        value.className = "batch-classes-chart-value";
        value.textContent = count;

        row.appendChild(label);
        row.appendChild(barWrapper);
        row.appendChild(value);

        chartEl.appendChild(row);
      });
  }

  openModal("batch-modal");
}
