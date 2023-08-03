from fastapi import FastAPI
from app.controllers import floor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(floor.router, prefix="/api/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1")
def root():
    return {"message": "Hello World"}