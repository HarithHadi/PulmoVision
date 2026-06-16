# routes/auth_route.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from db import supabase
from auth_utils import verify_password, create_token, get_current_user

router = APIRouter()

class LoginRequest(BaseModel):
    radiologist_id: str
    password: str

@router.post("/login")
def login(body: LoginRequest):
    # Find user by radiologist_id
    result = supabase.table("users") \
        .select("id, radiologist_id, password_hash, role") \
        .eq("radiologist_id", body.radiologist_id) \
        .single() \
        .execute()

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    user = result.data

    # Verify password
    if not verify_password(body.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Generate JWT
    token = create_token(
        user_id=user["id"],
        radiologist_id=user["radiologist_id"]
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "radiologist_id": user["radiologist_id"],
        "role": user["role"]
    }

@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return {
        "user_id": current_user["sub"],
        "radiologist_id": current_user["radiologist_id"]
    }