import sys
from pathlib import Path

# Must be first — ensures 'chatbot/' package resolves before
# backend/app/routes/chatbot.py can shadow it
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routes import reviews
from backend.app.routes import chatbot
from backend.app.routes.analytics import router as analytics_router

app = FastAPI()

app.include_router(reviews.router, prefix="/reviews", tags=["reviews"])
app.include_router(chatbot.router)
app.include_router(analytics_router)

@app.get("/")
def read_root():
    return {"message": "Mument API running"}

origins = [
    "http://localhost",
    "http://127.0.0.1",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)