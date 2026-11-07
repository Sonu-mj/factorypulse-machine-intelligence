"""FastAPI application for FactoryPulse industrial machine failure prediction service.

Provides REST endpoints for health checks, model metadata inspection, single machine
telemetry prediction, and vectorized batch inference.

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

from contextlib import asynccontextmanager
import logging
from typing import Final, List, Literal, Optional
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from src.prediction.predictor import FailurePredictor
from src.prediction.schemas import (
    BatchPredictionResult,
    BatchTelemetryInput,
    MANDATORY_PROTOTYPE_DISCLAIMER,
    MachineTelemetryInput,
    PredictionResult,
)

logger = logging.getLogger(__name__)

API_VERSION: Final[str] = "0.4.0"
API_TITLE: Final[str] = "FactoryPulse API"
API_DESCRIPTION: Final[str] = (
    "Industrial Machine Health & Failure-Risk Intelligence. "
    f"{MANDATORY_PROTOTYPE_DISCLAIMER}."
)


# =============================================================================
# API Response Schemas
# =============================================================================

class HealthResponse(BaseModel):
    """API health and service availability response."""

    status: Literal["healthy", "unhealthy"] = Field(
        ...,
        description="Overall API service health status.",
        examples=["healthy"],
    )
    model_available: bool = Field(
        ...,
        description="Indicates whether the underlying ML model pipeline is loaded and operational.",
        examples=[True],
    )
    version: str = Field(
        default=API_VERSION,
        description="API semantic version.",
        examples=[API_VERSION],
    )
    disclaimer: str = Field(
        default=MANDATORY_PROTOTYPE_DISCLAIMER,
        description="Mandatory prototype status disclaimer.",
    )


class ModelInfoResponse(BaseModel):
    """Metadata response regarding the approved ML model and decision threshold."""

    model_name: str = Field(
        ...,
        description="Identifier of the trained machine learning model.",
        examples=["RandomForestClassifier (balanced)"],
    )
    model_type: str = Field(
        ...,
        description="Underlying architecture and pipeline type.",
        examples=["scikit-learn Pipeline (ColumnTransformer + RandomForestClassifier)"],
    )
    selected_threshold: float = Field(
        ...,
        description="Recommended decision threshold applied for failure classification.",
        examples=[0.45],
    )
    approved_features: List[str] = Field(
        ...,
        description="The strict list of 6 approved physical sensor features.",
    )
    disclaimer: str = Field(
        default=MANDATORY_PROTOTYPE_DISCLAIMER,
        description="Mandatory prototype status disclaimer.",
    )


# =============================================================================
# Dependency Injection Provider
# =============================================================================

def get_predictor(request: Request) -> FailurePredictor:
    """Dependency provider that extracts the initialized FailurePredictor service.

    Args:
        request: FastAPI HTTP request instance.

    Returns:
        FailurePredictor: Active prediction service.

    Raises:
        HTTPException: 503 Service Unavailable if predictor failed to initialize.
    """
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction service is currently unavailable. Model failed to initialize.",
        )
    return predictor


# =============================================================================
# Application Factory
# =============================================================================

def create_app(predictor: Optional[FailurePredictor] = None) -> FastAPI:
    """Create and configure the FastAPI application instance.

    Allows injecting a pre-configured or mock FailurePredictor for testing,
    or safely auto-initializes the default FailurePredictor at startup.

    Args:
        predictor: Optional pre-configured FailurePredictor instance.

    Returns:
        FastAPI: Configured application instance.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifecycle manager to safely initialize the prediction service at startup."""
        if getattr(app.state, "predictor", None) is None:
            try:
                app.state.predictor = FailurePredictor()
                app.state.model_available = True
                logger.info("FailurePredictor successfully initialized on startup.")
            except Exception as exc:
                logger.error("Failed to initialize FailurePredictor on startup: %s", exc)
                app.state.predictor = None
                app.state.model_available = False
        yield

    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        lifespan=lifespan,
    )

    # Initialize app state attributes
    if predictor is not None:
        app.state.predictor = predictor
        app.state.model_available = True
    else:
        try:
            app.state.predictor = FailurePredictor()
            app.state.model_available = True
        except Exception as exc:
            logger.error("Initial FailurePredictor setup failed: %s", exc)
            app.state.predictor = None
            app.state.model_available = False

    # -------------------------------------------------------------------------
    # Route Handlers (Synchronous def for CPU-bound ML inference)
    # -------------------------------------------------------------------------

    @app.get(
        "/health",
        response_model=HealthResponse,
        summary="API Health Check",
        description="Verify service availability and confirm ML model pipeline readiness.",
    )
    def health_check(request: Request, response: Response) -> HealthResponse:
        """Return API and ML model availability status."""
        model_available = getattr(request.app.state, "model_available", False)
        if not model_available:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return HealthResponse(
                status="unhealthy",
                model_available=False,
                version=API_VERSION,
                disclaimer=MANDATORY_PROTOTYPE_DISCLAIMER,
            )

        return HealthResponse(
            status="healthy",
            model_available=True,
            version=API_VERSION,
            disclaimer=MANDATORY_PROTOTYPE_DISCLAIMER,
        )

    @app.get(
        "/model/info",
        response_model=ModelInfoResponse,
        summary="Model & Threshold Metadata",
        description="Obtain model name, architecture, selected decision threshold, and approved features.",
    )
    def get_model_info(
        predictor: FailurePredictor = Depends(get_predictor),
    ) -> ModelInfoResponse:
        """Query metadata directly from the FailurePredictor service."""
        info = predictor.get_service_info()
        return ModelInfoResponse(
            model_name=info["model_name"],
            model_type="scikit-learn Pipeline (ColumnTransformer + RandomForestClassifier)",
            selected_threshold=info["applied_threshold"],
            approved_features=info["approved_features"],
            disclaimer=info["disclaimer"],
        )

    @app.post(
        "/predict",
        response_model=PredictionResult,
        status_code=status.HTTP_200_OK,
        summary="Predict Machine Failure",
        description="Evaluate telemetry for a single machine and return structured failure risk.",
    )
    def predict_failure(
        telemetry: MachineTelemetryInput,
        predictor: FailurePredictor = Depends(get_predictor),
    ) -> PredictionResult:
        """Execute single-machine failure risk prediction."""
        try:
            return predictor.predict(telemetry)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error("Unexpected prediction failure: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while generating the prediction.",
            ) from exc

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResult,
        status_code=status.HTTP_200_OK,
        summary="Batch Machine Failure Prediction",
        description="Vectorized evaluation for a collection of industrial machine telemetry records.",
    )
    def predict_batch_failures(
        batch: BatchTelemetryInput,
        predictor: FailurePredictor = Depends(get_predictor),
    ) -> BatchPredictionResult:
        """Execute vectorized batch failure risk prediction."""
        if not batch.machines or len(batch.machines) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Telemetry batch cannot be empty.",
            )

        try:
            results = predictor.predict_batch(batch)
            failure_count = sum(r.predicted_class for r in results)

            return BatchPredictionResult(
                total_evaluated=len(results),
                predicted_failures_count=failure_count,
                applied_threshold=predictor.threshold,
                predictions=results,
                disclaimer=MANDATORY_PROTOTYPE_DISCLAIMER,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error("Unexpected batch prediction failure: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while generating batch predictions.",
            ) from exc

    return app


# Default ASGI application instance
app = create_app()
