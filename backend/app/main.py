from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import reviews
from app.routes import analytics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reviews.router, prefix="/reviews")
app.include_router(analytics.router, prefix="/analytics")

@app.get("/")
def root():
    return {"message": "Mument Backend Running"}