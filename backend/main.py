from fastapi import FastAPI
from fastapi.responses import FileResponse
from dependencies import models
from backend.routes.tb_route import router as tb_router
from backend.routes.diagnose_route import router as diagnose_router
from backend.routes.report_route import router as report_router
from models_def import load_all_models
import torch
from fastapi.middleware.cors import CORSMiddleware
from db import supabase
from routes.auth_route import router as auth_router
from routes.patient_route import router as patient_router
from routes.diagnosis_route import router as diagnosis_save_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # or your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Loading models into VRAM...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf, llama, tok = load_all_models(device)

    models.classifier = clf
    models.llama = llama
    models.tokenizer = tok
    models.load_tb_model()

    print("All models are ready for iference")

app.include_router(tb_router)
app.include_router(diagnose_router)
app.include_router(report_router)


@app.get("/")
def serve_html():
    return FileResponse("Rad_DINO.html")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.get("/pulmovision")
def serve_pulmovision():
    return FileResponse("PulmoVision.html")


app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(patient_router, prefix="/patients", tags=["patients"])
app.include_router(diagnosis_save_router, prefix="/diagnoses", tags=["diagnoses"])