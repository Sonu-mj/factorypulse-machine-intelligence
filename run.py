"""FactoryPulse — Unified Command-Line Interface.

Provides a single root entry point for running all core workflows:
- Starting the FastAPI prediction backend server
- Starting the Streamlit industrial monitoring dashboard
- Executing the complete automated pytest test suite
- Running data validation, EDA, feature engineering, baseline training, and threshold analysis

Usage:
    python run.py api [--host 127.0.0.1] [--port 8000] [--reload]
    python run.py dashboard [--port 8501]
    python run.py test [-v]
    python run.py validate
    python run.py eda
    python run.py features
    python run.py train
    python run.py threshold

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

import argparse
from pathlib import Path
import subprocess
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DISCLAIMER: str = (
    "EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT"
)


def print_banner(action_title: str):
    """Print standard operational banner."""
    print("=" * 75)
    print(f"🏭 FactoryPulse — {action_title}")
    print(f"⚠️  {DISCLAIMER}")
    print("=" * 75)


def run_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Launch the FastAPI REST API server via Uvicorn."""
    print_banner(f"FastAPI Prediction Server (http://{host}:{port})")
    import uvicorn
    uvicorn.run("src.api.app:app", host=host, port=port, reload=reload, log_level="info")


def run_dashboard(port: int = 8501):
    """Launch the Streamlit industrial monitoring dashboard."""
    print_banner(f"Streamlit Dashboard (http://localhost:{port})")
    dashboard_path = REPO_ROOT / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(port),
    ]
    subprocess.run(cmd)


def run_tests(args_list=None):
    """Execute the complete pytest automated test suite."""
    print_banner("Automated Pytest Suite Execution")
    import pytest
    extra_args = args_list or ["-v"]
    exit_code = pytest.main(extra_args)
    sys.exit(exit_code)


def run_validation():
    """Run data ingestion and validation check."""
    print_banner("Data Ingestion & Schema Validation")
    from src.data.loader import load_and_validate_data
    df, report = load_and_validate_data("data/raw/ai4i2020.csv")
    print(report.summary())
    print(f"\nDataFrame loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.")
    if not report.is_valid:
        sys.exit(1)


def run_eda():
    """Run exploratory data analysis pipeline and generate report & figures."""
    print_banner("Exploratory Data Analysis (EDA) Pipeline")
    from src.features.eda import run_eda_pipeline
    results = run_eda_pipeline()
    print(f"\nEDA completed successfully.")
    print(f"Report saved to: {results['markdown_report']}")
    print(f"Generated figures: {len(results['saved_figures'])}")


def run_features():
    """Run feature engineering, stratified splitting, and standard scaling."""
    print_banner("Feature Engineering & Preprocessing Pipeline")
    from src.features.pipeline import run_feature_pipeline
    results = run_feature_pipeline()
    print("\nFeature pipeline completed successfully.")
    print(f"Engineered features: {results['feature_names']}")
    print(f"Train split: {results['train_samples']} samples | Test split: {results['test_samples']} samples.")


def run_training():
    """Run baseline model training, evaluation, and artifact generation."""
    print_banner("Baseline Model Training & Evaluation")
    from src.training.pipeline import run_baseline_training
    results = run_baseline_training()
    print("\nBaseline training completed successfully.")
    print(f"Trained models: {list(results['saved_models'].keys())}")
    print(f"Comparison report: {results['markdown_report']}")


def run_threshold():
    """Run 5-fold cross-validation threshold tuning and test set evaluation."""
    print_banner("Random Forest Decision Threshold Optimization")
    from src.training.threshold_analysis import run_threshold_analysis
    results = run_threshold_analysis()
    rec = results["recommendation"]["recommended_threshold"]
    print(f"\nThreshold analysis completed successfully.")
    print(f"Recommended Decision Cutoff: {rec:.2f}")
    print(f"Threshold report: {results['markdown_report']}")


def main():
    """Parse CLI commands and route to corresponding execution function."""
    parser = argparse.ArgumentParser(
        description="FactoryPulse Industrial Machine Health Intelligence CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available FactoryPulse commands")

    # API Command
    api_parser = subparsers.add_parser("api", help="Start FastAPI REST prediction server")
    api_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    api_parser.add_argument("--reload", action="store_true", help="Enable code auto-reload")

    # Dashboard Command
    dash_parser = subparsers.add_parser("dashboard", help="Start Streamlit monitoring dashboard")
    dash_parser.add_argument("--port", type=int, default=8501, help="Port (default: 8501)")

    # Test Command
    subparsers.add_parser("test", help="Execute complete automated pytest test suite")

    # Pipeline Commands
    subparsers.add_parser("validate", help="Run data ingestion and validation check")
    subparsers.add_parser("eda", help="Run exploratory data analysis pipeline")
    subparsers.add_parser("features", help="Run feature engineering and scaling pipeline")
    subparsers.add_parser("train", help="Train and evaluate baseline classification models")
    subparsers.add_parser("threshold", help="Execute 5-fold CV decision threshold optimization")

    args, unknown = parser.parse_known_args()

    if args.command == "api":
        run_api(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "dashboard":
        run_dashboard(port=args.port)
    elif args.command == "test":
        run_tests(unknown or ["-v"])
    elif args.command == "validate":
        run_validation()
    elif args.command == "eda":
        run_eda()
    elif args.command == "features":
        run_features()
    elif args.command == "train":
        run_training()
    elif args.command == "threshold":
        run_threshold()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

