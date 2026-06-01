# -*- coding: utf-8 -*-
"""
Point d'entrée — lance le serveur FastAPI.
En local : http://localhost:8000
En production (Render) : le port est fourni par la variable d'environnement PORT.
"""
import sys, os
from pathlib import Path
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    port   = int(os.getenv("PORT", 8000))
    debug  = os.getenv("RENDER") is None  # reload uniquement en local

    uvicorn.run(
        "web.main:app",
        host="0.0.0.0",
        port=port,
        reload=debug,
        reload_dirs=["web"] if debug else None,
    )
