# routes/patient_route.py
from fastapi import APIRouter, Depends
from db import supabase
from auth_utils import get_current_user

router = APIRouter()

@router.get("/search")
def search_patients(name: str, current_user: dict = Depends(get_current_user)):
    result = supabase.table("patients") \
        .select("*") \
        .ilike("name", f"%{name}%") \
        .execute()
    return result.data