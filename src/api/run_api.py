"""CLI entry point for running the FactoryPulse FastAPI prediction service with Uvicorn.

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import uvicorn
from src.prediction.schemas import MANDATORY_PROTOTYPE_DISCLAIMER


def parse_args():
    """Parse command-line arguments for server configuration."""
    parser = argparse.ArgumentParser(
        description="FactoryPulse Industrial Machine Health Prediction API Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to listen on (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development code changes",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 70)
    print("FactoryPulse API — Industrial Machine Health & Failure-Risk Intelligence")
    print(MANDATORY_PROTOTYPE_DISCLAIMER)
    print(f"Starting server on http://{args.host}:{args.port}")
    print("=" * 70)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
