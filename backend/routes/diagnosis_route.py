# routes/diagnosis_route.py
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from typing import Optional
from db import supabase
from auth_utils import get_current_user
from config import XRAY_SAVE_DIR, HEATMAP_SAVE_DIR
import shutil
import uuid
import os

router = APIRouter()

@router.post("/save")
def save_diagnosis(
    patient_id: int = Form(...),
    tb_probability: float = Form(...),
    llama_diagnosis: Optional[str] = Form(None),
    radiologist_diagnosis: Optional[str] = Form(None),
    radiologist_notes: Optional[str] = Form(None),
    clinical_data: Optional[str] = Form(None),
    xray_file: Optional[UploadFile] = File(None),
    heatmap_file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    import json
    uid = str(uuid.uuid4())[:8]
    xray_path = None
    heatmap_path = None

    if xray_file:
        os.makedirs(XRAY_SAVE_DIR, exist_ok=True)
        xray_filename = f"patient{patient_id}_{uid}_xray.jpg"
        xray_path = f"{XRAY_SAVE_DIR}/{xray_filename}"
        with open(xray_path, "wb") as f:
            shutil.copyfileobj(xray_file.file, f)

    if heatmap_file:
        os.makedirs(HEATMAP_SAVE_DIR, exist_ok=True)
        heatmap_filename = f"patient{patient_id}_{uid}_heatmap.jpg"
        heatmap_path = f"{HEATMAP_SAVE_DIR}/{heatmap_filename}"
        with open(heatmap_path, "wb") as f:
            shutil.copyfileobj(heatmap_file.file, f)

    # Parse clinical data JSON string
    clinical_parsed = None
    if clinical_data:
        try:
            clinical_parsed = json.loads(clinical_data)
        except Exception:
            clinical_parsed = None

    result = supabase.table("diagnoses").insert({
        "patient_id": patient_id,
        "radiologist_id": current_user["radiologist_id"],
        "tb_probability": tb_probability,
        "xray_path": xray_path,
        "heatmap_path": heatmap_path,
        "llama_diagnosis": llama_diagnosis,
        "radiologist_diagnosis": radiologist_diagnosis,
        "radiologist_notes": radiologist_notes,
        "clinical_data": clinical_parsed,
        "status": "pending"
    }).execute()

    return result.data[0]

@router.get("/all")
def get_all_diagnoses(current_user: dict = Depends(get_current_user)):
    result = supabase.table("diagnoses") \
        .select("*, patients(name, age, sex, contact_number)") \
        .order("created_at", desc=True) \
        .execute()
    return result.data

@router.get("/image")
def get_image(path: str, current_user: dict = Depends(get_current_user)):
    print(f"Requested path: '{path}'")
    print(f"File exists: {os.path.exists(path)}")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@router.get("/patient/{patient_id}")
def get_patient_diagnoses(patient_id: int, current_user: dict = Depends(get_current_user)):
    result = supabase.table("diagnoses") \
        .select("*") \
        .eq("patient_id", patient_id) \
        .order("created_at", desc=True) \
        .execute()
    return result.data