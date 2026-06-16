# routes/diagnosis_route.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional
from db import supabase
from auth_utils import get_current_user

router = APIRouter()

class DiagnosisCreate(BaseModel):
    patient_id: int
    tb_probability: float
    xray_path: Optional[str] = None
    heatmap_path: Optional[str] = None
    llama_diagnosis: Optional[str] = None
    radiologist_diagnosis: Optional[str] = None
    radiologist_notes: Optional[str] = None
    status: str = "pending"

@router.post("/save")
def save_diagnosis(body: DiagnosisCreate, current_user: dict = Depends(get_current_user)):
    result = supabase.table("diagnoses").insert({
        "patient_id": body.patient_id,
        "radiologist_id": current_user["radiologist_id"],
        "tb_probability": body.tb_probability,
        "xray_path": body.xray_path,
        "heatmap_path": body.heatmap_path,
        "llama_diagnosis": body.llama_diagnosis,
        "radiologist_diagnosis": body.radiologist_diagnosis,
        "radiologist_notes": body.radiologist_notes,
        "status": body.status
    }).execute()
    return result.data

@router.get("/patient/{patient_id}")
def get_patient_diagnoses(patient_id: int, current_user: dict = Depends(get_current_user)):
    result = supabase.table("diagnoses") \
        .select("*") \
        .eq("patient_id", patient_id) \
        .order("created_at", desc=True) \
        .execute()
    return result.data