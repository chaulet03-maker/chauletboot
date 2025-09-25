import asyncio
from fastapi import FastAPI
import uvicorn

app = FastAPI()
@app.get("/healthz")
def healthz():
    return {"ok": True}

async def start_health_server(config):
    port = int(config.get("health",{}).get("http_port",8080))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()
