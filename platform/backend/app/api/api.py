from app.api.endpoints import auth, signals, stream, trades

api_router = APIRouter()
api_router.include_router(auth.router, tags=["login"])
api_router.include_router(signals.router, prefix="/signals", tags=["signals"])
api_router.include_router(trades.router, prefix="/trades", tags=["trades"])
api_router.include_router(stream.router, tags=["stream"])
