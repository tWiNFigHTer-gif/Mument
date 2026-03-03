from fastapi import FastAPI
from app.routes import reviews
from app.routes import analytics

# Create FastAPI instance FIRST
app = FastAPI()

# Include routers AFTER creating app
app.include_router(reviews.router, prefix="/reviews")
app.include_router(analytics.router, prefix="/analytics")

@app.get("/")
def root():
    return {"message": "Mument Backend Running"}