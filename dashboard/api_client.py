"""FastAPI client for the FactoryPulse Streamlit Dashboard.

Handles HTTP communication with the FactoryPulse REST API service, providing structured
methods for health checks, model metadata inspection, and failure risk prediction.
Gracefully handles connection drops, timeouts, validation errors, and server errors.

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, Optional
import requests

DEFAULT_API_URL: str = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT: float = 5.0
DEFAULT_HEALTH_TIMEOUT: float = 2.5


@dataclass
class APIResponse:
    """Structured response container for API calls."""

    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None


class FactoryPulseAPIClient:
    """HTTP client interface for FactoryPulse prediction and health endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize the API client.

        Args:
            base_url: Base URL of the FastAPI service. Defaults to FACTORYPULSE_API_URL
                      environment variable or 'http://127.0.0.1:8000'.
            timeout: Default request timeout in seconds (default: 5.0s).
        """
        env_url = os.getenv("FACTORYPULSE_API_URL", DEFAULT_API_URL).rstrip("/")
        self.base_url = (base_url or env_url).rstrip("/")
        self.timeout = float(timeout)

    def check_health(self) -> APIResponse:
        """Query the /health endpoint to verify service and model availability.

        Returns:
            APIResponse: Status outcome, parsed payload, and status code.
        """
        url = f"{self.base_url}/health"
        try:
            resp = requests.get(url, timeout=DEFAULT_HEALTH_TIMEOUT)
            status_code = resp.status_code

            try:
                payload = resp.json()
            except Exception:
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message="Health check returned non-JSON response.",
                )

            if status_code == 200 and payload.get("model_available") is True:
                return APIResponse(
                    success=True,
                    data=payload,
                    status_code=status_code,
                )
            elif status_code == 503 or payload.get("model_available") is False:
                return APIResponse(
                    success=False,
                    data=payload,
                    status_code=status_code,
                    error_message="Backend unavailable (ML model pipeline is not ready).",
                )
            else:
                return APIResponse(
                    success=False,
                    data=payload,
                    status_code=status_code,
                    error_message=f"Health check failed with HTTP {status_code}.",
                )

        except requests.exceptions.ConnectionError:
            return APIResponse(
                success=False,
                error_message=f"Disconnected: Could not reach FastAPI server at {self.base_url}.",
            )
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error_message=f"Health check timed out after {DEFAULT_HEALTH_TIMEOUT} seconds.",
            )
        except requests.exceptions.RequestException as exc:
            return APIResponse(
                success=False,
                error_message=f"Network error during health check: {exc}",
            )

    def get_model_info(self) -> APIResponse:
        """Retrieve approved model metadata and decision threshold from /model/info.

        Returns:
            APIResponse: Model metadata dictionary on success.
        """
        url = f"{self.base_url}/model/info"
        try:
            resp = requests.get(url, timeout=self.timeout)
            status_code = resp.status_code

            try:
                payload = resp.json()
            except Exception:
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message="Model info endpoint returned non-JSON response.",
                )

            if status_code == 200:
                return APIResponse(success=True, data=payload, status_code=status_code)
            else:
                detail = payload.get("detail", f"HTTP {status_code}")
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"Failed to fetch model info: {detail}",
                )

        except requests.exceptions.ConnectionError:
            return APIResponse(
                success=False,
                error_message=f"Connection refused: Unable to reach {self.base_url}.",
            )
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error_message=f"Model info request timed out after {self.timeout}s.",
            )
        except requests.exceptions.RequestException as exc:
            return APIResponse(
                success=False,
                error_message=f"Network error: {exc}",
            )

    def predict(self, telemetry: Dict[str, Any]) -> APIResponse:
        """Submit machine telemetry to /predict and return structured failure assessment.

        Args:
            telemetry: Dictionary containing the 6 approved sensor features:
                       type, air_temperature_k, process_temperature_k,
                       rotational_speed_rpm, torque_nm, tool_wear_min,
                       and optional machine_id.

        Returns:
            APIResponse: Prediction result payload or descriptive error message.
        """
        if not isinstance(telemetry, dict) or not telemetry:
            return APIResponse(
                success=False,
                error_message="Telemetry input must be a non-empty dictionary.",
            )

        url = f"{self.base_url}/predict"
        try:
            resp = requests.post(url, json=telemetry, timeout=self.timeout)
            status_code = resp.status_code

            try:
                payload = resp.json()
            except Exception:
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message="Prediction service returned a malformed, non-JSON response.",
                )

            if status_code == 200:
                # Validate that essential prediction attributes exist in the response
                required_fields = ["failure_probability", "applied_threshold", "predicted_class", "prediction_label"]
                missing = [f for f in required_fields if f not in payload]
                if missing:
                    return APIResponse(
                        success=False,
                        status_code=status_code,
                        error_message=f"Malformed API response: missing required fields {missing}.",
                    )

                return APIResponse(success=True, data=payload, status_code=status_code)

            elif status_code == 422:
                # Pydantic validation error format
                details = payload.get("detail", [])
                if isinstance(details, list) and details:
                    messages = []
                    for err in details:
                        loc = " -> ".join(str(item) for item in err.get("loc", []))
                        msg = err.get("msg", "invalid")
                        messages.append(f"{loc}: {msg}")
                    formatted_error = "; ".join(messages)
                else:
                    formatted_error = str(details)

                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"Validation Error (HTTP 422): {formatted_error}",
                )

            elif status_code == 400:
                detail = payload.get("detail", "Bad Request")
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"Input Error (HTTP 400): {detail}",
                )

            elif status_code == 503:
                detail = payload.get("detail", "Service Unavailable")
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"Backend Unavailable (HTTP 503): {detail}",
                )

            elif status_code == 500:
                detail = payload.get("detail", "Internal Server Error")
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"API Server Error (HTTP 500): {detail}",
                )

            else:
                detail = payload.get("detail", f"Unexpected status HTTP {status_code}")
                return APIResponse(
                    success=False,
                    status_code=status_code,
                    error_message=f"API Error (HTTP {status_code}): {detail}",
                )

        except requests.exceptions.ConnectionError:
            return APIResponse(
                success=False,
                error_message=(
                    f"Connection Refused: Cannot reach FactoryPulse API at {self.base_url}. "
                    "Ensure the FastAPI service is running."
                ),
            )
        except requests.exceptions.Timeout:
            return APIResponse(
                success=False,
                error_message=f"Prediction Request Timed Out: Server did not respond within {self.timeout}s.",
            )
        except requests.exceptions.RequestException as exc:
            return APIResponse(
                success=False,
                error_message=f"Network error while connecting to prediction API: {exc}",
            )

