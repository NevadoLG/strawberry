import os
import sys
import base64
from collections import defaultdict
from typing import List

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_DIR, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from core.model import Config, SGSNet, get_transforms, non_max_suppression 

app = FastAPI(title="Strawberry Vision API")

device = Config.DEVICE

MODEL_PATH = os.path.join(
    "src", "data", "processed", "models", "best_model.pth"
)

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")

model = SGSNet(Config.NUM_CLASSES).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

_, val_transform = get_transforms()

CLASS_COLORS = {
    "flowering": (255, 0, 255),   
    "growing_g": (0, 255, 0),    
    "growing_w": (0, 255, 255),   
    "nearly_m": (0, 165, 255),   
    "mature": (0, 0, 255),   
}

def run_inference(image_bytes: bytes, conf_threshold: float = 0.4):

    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    orig_h, orig_w = original_image.shape[:2]

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    transformed = val_transform(image=image_rgb, bboxes=[], class_labels=[])
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    batch_detections = non_max_suppression(
        predictions,
        conf_threshold=conf_threshold,
        iou_threshold=Config.IOU_THRESHOLD,
    )

    boxes, labels, scores = batch_detections[0]  

    detections: List[dict] = []
    for box, label, score in zip(boxes, labels, scores):
        cx, cy, w, h = box.tolist()
        class_idx = int(label.item())
        class_name = Config.CLASS_NAMES[class_idx]

        detections.append(
            {
                "center": {"cx": float(cx), "cy": float(cy)},
                "size": {"w": float(w), "h": float(h)},
                "class_index": class_idx,
                "class_name": class_name,
                "score": float(score.item()),
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    max_to_show = 30
    detections = detections[:max_to_show]

    summary_by_class = defaultdict(lambda: {"count": 0, "best_score": 0.0})
    for det in detections:
        name = det["class_name"]
        summary_by_class[name]["count"] += 1
        summary_by_class[name]["best_score"] = max(
            summary_by_class[name]["best_score"], det["score"]
        )

    if detections:
        max_conf = max(d["score"] for d in detections)
        mean_conf = float(sum(d["score"] for d in detections) / len(detections))
    else:
        max_conf = 0.0
        mean_conf = 0.0

    return {
        "image_size": {"width": orig_w, "height": orig_h},
        "num_detections": len(detections),
        "max_confidence": float(max_conf),
        "mean_confidence": float(mean_conf),
        "detections": detections,
        "summary_by_class": summary_by_class,
    }

@app.get("/")
def root():
    return {"mensaje": "API de Strawberry Vision funcionando correctamente"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen, ejecuta el modelo, dibuja las bounding boxes
    sobre la imagen original y devuelve:
      - imagen anotada (base64)
      - lista de detecciones
      - resumen por clase
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen")

    orig_h, orig_w = img.shape[:2]

    result = run_inference(contents, conf_threshold=Config.CONF_THRESHOLD)
    detections = result["detections"]

    for det in detections:
        cx = det["center"]["cx"]
        cy = det["center"]["cy"]
        w_rel = det["size"]["w"]
        h_rel = det["size"]["h"]
        class_name = det["class_name"]
        score = det["score"]

        x1 = int((cx - w_rel / 2) * orig_w)
        y1 = int((cy - h_rel / 2) * orig_h)
        x2 = int((cx + w_rel / 2) * orig_w)
        y2 = int((cy + h_rel / 2) * orig_h)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        color = CLASS_COLORS.get(class_name, (0, 255, 0))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} ({score:.2f})"
        cv2.putText(
            img,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if not detections:
        cv2.putText(
            img,
            "Sin detecciones confiables",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    _, buffer = cv2.imencode(".png", img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    summary_by_class = {
        k: {"count": v["count"], "best_score": v["best_score"]}
        for k, v in result["summary_by_class"].items()
    }

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "annotated_image": f"data:image/png;base64,{img_base64}",
        "mensaje": "Imagen procesada con el modelo (detecciones TOP)",
        "num_detections": result["num_detections"],
        "summary_by_class": summary_by_class,
        "detections": detections,
    }

@app.get("/ui", response_class=HTMLResponse)
def ui():
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8" />
        <title>Strawberry Vision – Demo</title>
        <style>
            body {
                font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: #020617;
                color: #e5e7eb;
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 24px;
            }
            h1 {
                margin-bottom: 8px;
                font-size: 32px;
            }
            p.subtitle {
                margin: 0;
                opacity: 0.8;
            }
            .panel {
                display: flex;
                flex-wrap: wrap;
                gap: 24px;
                margin-top: 28px;
                max-width: 1120px;
                width: 100%;
                justify-content: center;
            }
            .card {
                background: #020617;
                border-radius: 16px;
                padding: 18px;
                box-shadow: 0 18px 40px rgba(15,23,42,0.9);
                width: 420px;
                text-align: center;
                border: 1px solid rgba(148,163,184,0.25);
            }
            .card h2 {
                margin-top: 0;
                margin-bottom: 8px;
                font-size: 20px;
            }
            img {
                max-width: 100%;
                border-radius: 12px;
                margin-top: 10px;
                background: #0b1120;
            }
            button {
                margin-top: 10px;
                padding: 10px 20px;
                border-radius: 9999px;
                border: none;
                background: #22c55e;
                color: #020617;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.1s ease;
            }
            button:hover {
                transform: translateY(-1px);
                box-shadow: 0 10px 20px rgba(34,197,94,0.35);
                background: #16a34a;
            }
            button:disabled {
                opacity: 0.5;
                cursor: not-allowed;
                box-shadow: none;
                transform: none;
            }
            input[type="file"] {
                margin-top: 10px;
            }
            .msg {
                margin-top: 8px;
                font-size: 0.9rem;
                color: #9ca3af;
            }
            .legend {
                margin-top: 20px;
                padding: 12px 16px;
                border-radius: 12px;
                background: rgba(15,23,42,0.9);
                border: 1px solid rgba(148,163,184,0.35);
                font-size: 0.9rem;
                max-width: 860px;
            }
            .legend-title {
                font-weight: 600;
                margin-bottom: 8px;
            }
            .legend-row {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .tag {
                display: inline-flex;
                align-items: center;
                gap: 6px;
                padding: 4px 10px;
                border-radius: 9999px;
                background: #0b1120;
            }
            .dot {
                width: 10px;
                height: 10px;
                border-radius: 9999px;
            }
            .dot.flowering { background: #ec4899; }   /* ~magenta */
            .dot.growing_g { background: #22c55e; }   /* verde */
            .dot.growing_w { background: #22d3ee; }   /* cyan */
            .dot.nearly_m { background: #f97316; }    /* naranja */
            .dot.mature   { background: #ef4444; }    /* rojo */
            .summary {
                margin-top: 14px;
                text-align: left;
                font-size: 0.9rem;
                max-height: 180px;
                overflow-y: auto;
            }
            .summary h3 {
                margin: 0 0 6px 0;
                font-size: 0.95rem;
            }
            .summary ul {
                margin: 0;
                padding-left: 18px;
            }
        </style>
    </head>
    <body>
        <h1>Strawberry Vision – Demo de Inferencia</h1>
        <p class="subtitle">Sube una imagen de fresas, el modelo detecta cada fruto y marca su estado de madurez.</p>

        <div class="panel">
            <div class="card">
                <h2>Imagen de entrada</h2>
                <input id="fileInput" type="file" accept="image/*" />
                <button id="btnSend" disabled>Enviar al modelo</button>
                <p class="msg" id="statusMsg">Selecciona una imagen para comenzar.</p>
                <img id="inputPreview" alt="Previsualización" />
            </div>

            <div class="card">
                <h2>Salida del modelo</h2>
                <p class="msg">Cada recuadro coloreado corresponde a una fresa detectada.</p>
                <img id="outputImage" alt="Salida del modelo" />
                <div class="summary">
                    <h3>Resumen de detecciones</h3>
                    <ul id="summaryList"></ul>
                </div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-title">Estados de madurez</div>
            <div class="legend-row">
                <div class="tag"><span class="dot flowering"></span> flowering</div>
                <div class="tag"><span class="dot growing_g"></span> growing_g</div>
                <div class="tag"><span class="dot growing_w"></span> growing_w</div>
                <div class="tag"><span class="dot nearly_m"></span> nearly_m</div>
                <div class="tag"><span class="dot mature"></span> mature</div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById("fileInput");
            const btnSend = document.getElementById("btnSend");
            const statusMsg = document.getElementById("statusMsg");
            const inputPreview = document.getElementById("inputPreview");
            const outputImage = document.getElementById("outputImage");
            const summaryList = document.getElementById("summaryList");

            let selectedFile = null;

            fileInput.addEventListener("change", (event) => {
                const file = event.target.files[0];
                selectedFile = file || null;

                summaryList.innerHTML = "";

                if (!file) {
                    btnSend.disabled = true;
                    inputPreview.src = "";
                    outputImage.src = "";
                    statusMsg.textContent = "Selecciona una imagen para comenzar.";
                    return;
                }

                const reader = new FileReader();
                reader.onload = (e) => {
                    inputPreview.src = e.target.result;
                };
                reader.readAsDataURL(file);

                btnSend.disabled = false;
                statusMsg.textContent = "Imagen lista. Haz clic en 'Enviar al modelo'.";
            });

            btnSend.addEventListener("click", async () => {
                if (!selectedFile) return;

                btnSend.disabled = true;
                statusMsg.textContent = "Procesando imagen...";

                summaryList.innerHTML = "";

                const formData = new FormData();
                formData.append("file", selectedFile);

                try {
                    response = await fetch("http://localhost:8000/predict", {
                        method: "POST",
                        body: formData,
                    });

                    if (!response.ok) {
                        statusMsg.textContent = "Error en la inferencia.";
                        btnSend.disabled = false;
                        return;
                    }

                    const data = await response.json();
                    statusMsg.textContent = data.mensaje || "Inferencia realizada.";

                    if (data.annotated_image) {
                        outputImage.src = data.annotated_image;
                    }

                    if (data.summary_by_class) {
                        const entries = Object.entries(data.summary_by_class);
                        if (!entries.length) {
                            summaryList.innerHTML = "<li>Sin detecciones confiables.</li>";
                        } else {
                            entries.forEach(([name, info]) => {
                                const li = document.createElement("li");
                                const count = info.count ?? info["count"];
                                const best = info.best_score ?? info["best_score"];
                                li.textContent = `${name}: ${count} detección(es), mejor confianza ${(best * 100).toFixed(1)}%`;
                                summaryList.appendChild(li);
                            });
                        }
                    }
                } catch (err) {
                    console.error(err);
                    statusMsg.textContent = "Error de red o del servidor.";
                } finally {
                    btnSend.disabled = false;
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)