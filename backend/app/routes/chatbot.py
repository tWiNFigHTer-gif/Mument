# Chatbot routes
import sys
from pathlib import Path

# Insert project-root at the front of sys.path so 'chatbot' resolves to
# the chatbot/ package, not this file (backend/app/routes/chatbot.py)
_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import APIRouter
from pydantic import BaseModel

from chatbot.engine.rules import generate_response

router = APIRouter(prefix="/chatbot", tags=["chatbot"])


class ChatRequest(BaseModel):
    message: str


@router.post("/ask")
def ask_chatbot(req: ChatRequest):
    result = generate_response(req.message)
    return result