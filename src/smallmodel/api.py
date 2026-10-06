import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .auth import configured_keys, identity


class TextBatch(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)


def create_app(artifact=None, keys=None):
    @asynccontextmanager
    async def lifespan(app):
        if not app.state.keys:
            raise RuntimeError("Set API_KEYS_JSON or APP_DEMO=1")
        path = artifact or os.getenv("MODEL_ARTIFACT")
        app.state.engine = None
        if path:
            from .inference import IntentEngine

            app.state.engine = IntentEngine(path, os.getenv("MODEL_INT8") == "1")
        yield

    app = FastAPI(title="Banking intent lab", lifespan=lifespan)
    app.state.keys = configured_keys() if keys is None else keys
    root = Path(__file__).parent
    app.mount("/assets", StaticFiles(directory=root / "web"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(root / "web/index.html")

    @app.get("/app-config")
    def config():
        return {"demo": os.getenv("APP_DEMO") == "1", "model_ready": app.state.engine is not None}

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": app.state.engine is not None}

    @app.get("/study")
    def study():
        return json.loads((root / "resources/banking77.json").read_text())

    @app.get("/model")
    def model(actor=Depends(identity)):
        engine = app.state.engine
        return {
            "ready": bool(engine),
            "sha256": engine.fingerprint if engine else None,
            "quantized": engine.quantized if engine else None,
            "metadata": engine.metadata if engine else None,
        }

    @app.post("/predict")
    def predict(batch: TextBatch, actor=Depends(identity)):
        if app.state.engine is None:
            raise HTTPException(503, "Load a released artifact with --artifact or --download-model")
        try:
            return app.state.engine.predict(batch.texts)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    return app


app = create_app()
