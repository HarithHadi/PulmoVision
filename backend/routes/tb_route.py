"""
FastAPI router — RAD-DINO TB inference + GradCAM overlay

FIXES vs original:
    - forward hook does NOT call .detach() on activations
    - GradCAM formula corrected: channel_weights = grads.mean(dim=0)
    - activation/gradient buffers are cleared before each call
"""


from fastapi import APIRouter, Depends, File, UploadFile
from dependencies import get_tb_model


router = APIRouter()


@router.post("/diagnoseTB")
async def diagnose_tb(
    file: UploadFile = File(...), 
    tb_model = Depends(get_tb_model) # Inject the model
):content = await file.read()
