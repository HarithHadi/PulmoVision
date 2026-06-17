# routes/patient_route.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from db import supabase
from auth_utils import get_current_user

router = APIRouter()

class PatientCreate(BaseModel):
    name: str
    sex: str
    age: int
    contact_number: Optional[str] = None

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    contact_number: Optional[str] = None

# Search patients by name (for autofill)
@router.get("/search")
def search_patients(name: str, current_user: dict = Depends(get_current_user)):
    result = supabase.table("patients") \
        .select("*") \
        .ilike("name", f"%{name}%") \
        .limit(5) \
        .execute()
    return result.data

# Get single patient by ID
@router.get("/{patient_id}")
def get_patient(patient_id: int, current_user: dict = Depends(get_current_user)):
    result = supabase.table("patients") \
        .select("*") \
        .eq("id", patient_id) \
        .single() \
        .execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Patient not found")
    return result.data

# Create new patient
@router.post("/")
def create_patient(body: PatientCreate, current_user: dict = Depends(get_current_user)):
    result = supabase.table("patients").insert({
        "name": body.name,
        "sex": body.sex,
        "age": body.age,
        "contact_number": body.contact_number
    }).execute()
    return result.data[0]

# Update existing patient
@router.patch("/{patient_id}")
def update_patient(patient_id: int, body: PatientUpdate, current_user: dict = Depends(get_current_user)):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = supabase.table("patients") \
        .update(updates) \
        .eq("id", patient_id) \
        .execute()
    return result.data[0]