import base64
from typing import Any, Dict

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse

from inference_service import infer_from_bytes

app = FastAPI(title="Strawberry Vision API")


@app.get("/")
def healthcheck() -> Dict[str, str]:
    """
    Endpoint simple de salud. Para revisión rápida: docker, uvicorn, etc.
    """
    return {"status": "ok", "message": "API de Strawberry Vision funcionando"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(
        0.4,
        ge=0.05,
        le=0.9,
        description="Umbral de confianza para filtrar detecciones",
    ),
) -> JSONResponse:
    """
    Recibe una imagen subida por el cliente (multipart/form-data),
    ejecuta el modelo y devuelve:
      - metadatos de la imagen
      - lista de detecciones
      - resumen por clase
      - imagen anotada en base64 (formato PNG)
    """
    try:
        contents = await file.read()
        result = infer_from_bytes(contents, conf_threshold=conf)

        annotated_bgr = result.pop("annotated_image_bgr")
        success, buffer = cv2.imencode(".png", annotated_bgr)
        if not success:
            raise RuntimeError("No se pudo codificar la imagen anotada.")

        image_base64 = base64.b64encode(buffer).decode("utf-8")
        result["annotated_image_base64"] = image_base64

        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al procesar la imagen: {e}",
        ) from e


@app.get("/demo", response_class=HTMLResponse)
def demo_page() -> str:
    """
    Página HTML mínima para probar el endpoint /predict.
    """
    html = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="utf-8" />
        <title>Strawberry Vision - Demo</title>
    </head>
    <body>
        <h1>Demo rápida - Strawberry Vision</h1>
        <form id="upload-form">
            <input type="file" id="file-input" accept="image/*" required />
            <button type="submit">Analizar imagen</button>
        </form>
        <p id="status"></p>
        <img id="result-image" style="max-width: 500px; display: none;" />
        <pre id="json-output" style="white-space: pre-wrap;"></pre>

        <script>
        const form = document.getElementById("upload-form");
        const fileInput = document.getElementById("file-input");
        const status = document.getElementById("status");
        const img = document.getElementById("result-image");
        const pre = document.getElementById("json-output");

        form.addEventListener("submit", async (e) => {
            e.preventDefault();
            if (!fileInput.files.length) {
                alert("Selecciona una imagen primero.");
                return;
            }
            status.textContent = "Enviando imagen...";
            img.style.display = "none";
            pre.textContent = "";

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            try {
                const resp = await fetch("/predict?conf=0.7", {
                    method: "POST",
                    body: formData
                });
                const data = await resp.json();
                if (!resp.ok) {
                    status.textContent = "Error: " + (data.detail || "desconocido");
                    return;
                }
                status.textContent = "OK";

                if (data.annotated_image_base64) {
                    img.src = "data:image/png;base64," + data.annotated_image_base64;
                    img.style.display = "block";
                }
                const copy = Object.assign({}, data);
                delete copy.annotated_image_base64;
                pre.textContent = JSON.stringify(copy, null, 2);
            } catch (err) {
                console.error(err);
                status.textContent = "Error de red o del servidor.";
            }
        });
        </script>
    </body>
    </html>
    """
    return html
