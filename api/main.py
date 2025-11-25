import os
import base64
from typing import Dict

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from inference_service import infer_from_bytes

# =========================
# RUTAS DE PROYECTO
# =========================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(ROOT_DIR, "templates")
STATIC_DIR = os.path.join(ROOT_DIR, "static")

app = FastAPI(title="Strawberry Vision API")

# Archivos estáticos (CSS, JS, imágenes)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Sistema de templates (Jinja2)
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# =========================
# ENDPOINTS HTML (PÁGINAS)
# =========================

@app.get("/", response_class=HTMLResponse, name="home")
def home(request: Request):
    # IMPORTANTE: usamos index.html, no home.html
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.get("/single", response_class=HTMLResponse, name="single_image_page")
def single_image_page(request: Request):
    return templates.TemplateResponse(
        "single_image.html",
        {"request": request},
    )


@app.get("/batch", response_class=HTMLResponse, name="batch_page")
def batch_page(request: Request):
    return templates.TemplateResponse(
        "batch.html",
        {"request": request},
    )


@app.get("/results", response_class=HTMLResponse, name="results_page")
def results_page(request: Request):
    return templates.TemplateResponse(
        "results.html",
        {"request": request},
    )


@app.get("/about", response_class=HTMLResponse, name="about_page")
def about_page(request: Request):
    return templates.TemplateResponse(
        "about.html",
        {"request": request},
    )

@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok", "message": "API de Strawberry Vision funcionando"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.05, le=0.9, description="Umbral de confianza"),
) -> JSONResponse:
    """
    Recibe una imagen, ejecuta el modelo y devuelve:
      - metadatos de la imagen
      - lista de detecciones
      - resumen por clase
      - imagen anotada en base64 (PNG)
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
