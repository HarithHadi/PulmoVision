from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from tb_route import router as tb_router
app.include_router(tb_router)

from diagnose_route import router as diagnose_router
app.include_router(diagnose_router)

@app.get("/")
def serve_html():
    return FileResponse("Rad_DINO.html")

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


from report_route import router as report_router
app.include_router(report_router)


@app.get("/pulmovision")
def serve_pulmovision():
    return FileResponse("PulmoVision.html")