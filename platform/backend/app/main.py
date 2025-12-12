from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.api import api_router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

app.include_router(api_router, prefix=settings.API_V1_STR)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import asyncio

from app.services.streamer import streamer


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(streamer.start())


@app.on_event("shutdown")
def shutdown_event():
    streamer.stop()


@app.get("/")
def root():
    return {"message": "Welcome to Crypto AI Platform API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}
