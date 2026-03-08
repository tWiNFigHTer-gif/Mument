from fastapi import FastAPI
from backend.app.routes import reviews

app = FastAPI()

app.include_router(reviews.router)

@app.get("/")
def read_root():
    return {"message": "Mument API running"}
