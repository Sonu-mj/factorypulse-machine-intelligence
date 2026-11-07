"""FactoryPulse — Industrial Machine Health & Failure-Risk Intelligence Dashboard.

Phase 5.2: Complete FastAPI Prediction Service Integration
Integrates with the FactoryPulse REST API backend (/health, /model/info, /predict),
providing an interactive operator form, real-time connection status monitoring,
and live Random Forest failure prediction results with session state persistence.

EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT.
"""

import os
from typing import Any, Dict, Optional
import streamlit as st

from dashboard.api_client import FactoryPulseAPIClient

# Mandatory prototype disclaimer
DISCLAIMER_TEXT = "EXPERIMENTAL BENCHMARK PROTOTYPE — NOT VALIDATED FOR REAL INDUSTRIAL DEPLOYMENT"

# The 6 approved model features
APPROVED_FEATURES = [
    {"name": "Type", "unit": "Variant", "description": "Machine quality variant (L: Low, M: Medium, H: High)"},
    {"name": "Air temperature [K]", "unit": "Kelvin (K)", "description": "Ambient environmental temperature"},
    {"name": "Process temperature [K]", "unit": "Kelvin (K)", "description": "Cutting point process temperature"},
    {"name": "Rotational speed [rpm]", "unit": "RPM", "description": "Spindle motor rotation speed"},
    {"name": "Torque [Nm]", "unit": "Newton-meters (Nm)", "description": "Spindle motor torque output"},
    {"name": "Tool wear [min]", "unit": "Minutes", "description": "Cumulative cutter tool wear time"},
]


# =============================================================================
# API Client & Health Check Helpers
# =============================================================================

@st.cache_resource
def get_api_client() -> FactoryPulseAPIClient:
    """Instantiate and cache the FactoryPulse API client."""
    api_url = os.getenv("FACTORYPULSE_API_URL", "http://127.0.0.1:8000")
    return FactoryPulseAPIClient(base_url=api_url, timeout=5.0)


@st.cache_data(ttl=10)
def check_cached_health() -> Dict[str, Any]:
    """Check API health with a 10-second cache to prevent excessive network requests.

    Returns:
        dict: Serialized health check outcome with status, connected flag, and message.
    """
    client = get_api_client()
    resp = client.check_health()
    if resp.success:
        return {
            "status": "Connected",
            "connected": True,
            "message": "FastAPI service operational and model pipeline ready.",
            "data": resp.data,
        }
    elif resp.status_code == 503:
        return {
            "status": "Backend Unavailable",
            "connected": False,
            "message": resp.error_message or "API reachable but ML model pipeline failed to initialize.",
            "data": resp.data,
        }
    else:
        return {
            "status": "Disconnected",
            "connected": False,
            "message": resp.error_message or f"Unable to reach API server at {client.base_url}.",
            "data": None,
        }


# =============================================================================
# Layout & Header Components
# =============================================================================

def render_header():
    """Render the dashboard top branding, subtitle, and prototype disclaimer."""
    st.title("🏭 FactoryPulse")
    st.subheader("Industrial Machine Health & Failure-Risk Intelligence")
    st.caption(f"⚠️ **Disclaimer:** {DISCLAIMER_TEXT}")
    st.markdown("---")


def render_sidebar() -> str:
    """Render sidebar navigation, backend connection status, and system info.

    Returns:
        str: Selected navigation page.
    """
    st.sidebar.title("⚙️ FactoryPulse")
    st.sidebar.markdown("**System Navigation**")

    # Main Navigation Radio
    page = st.sidebar.radio(
        "Go to view:",
        options=["Overview", "Machine Monitoring", "Failure Prediction"],
        index=2,  # Default to Failure Prediction for quick testing
    )

    st.sidebar.markdown("---")

    # Real-time API Connection Status
    st.sidebar.markdown("### 🔌 Backend Connection")
    client = get_api_client()
    st.sidebar.caption(f"Target API: `{client.base_url}`")

    # Optional manual refresh button
    if st.sidebar.button("🔄 Check Connection", help="Refresh API connection status immediately"):
        check_cached_health.clear()

    health = check_cached_health()
    if health["status"] == "Connected":
        st.sidebar.success("🟢 **Connected**\n\nFastAPI & Model Operational")
    elif health["status"] == "Backend Unavailable":
        st.sidebar.warning("🟡 **Backend Unavailable**\n\nAPI online, model not loaded")
    else:
        st.sidebar.error("🔴 **Disconnected**\n\nFastAPI server unreachable")
        st.sidebar.info("To start the backend server:\n```powershell\npython src/api/run_api.py\n```")

    st.sidebar.markdown("---")

    # Approved Feature Reference
    with st.sidebar.expander("ℹ️ Approved Features (6)"):
        for feat in APPROVED_FEATURES:
            st.markdown(f"- **{feat['name']}** ({feat['unit']})")

    st.sidebar.caption(f"Prototype Status:\n{DISCLAIMER_TEXT}")

    return page


# =============================================================================
# 1. Overview Section
# =============================================================================

def render_overview_section():
    """Render Machine Overview with system health and architecture summary."""
    st.header("📋 Machine Overview")
    st.write(
        "High-level telemetry and operating summary for connected industrial machining units."
    )

    health = check_cached_health()
    client = get_api_client()

    # Metric Row using st.columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="API Gateway Status",
            value="Online" if health["connected"] else "Offline",
            delta="Active" if health["connected"] else "Inactive",
            delta_color="normal" if health["connected"] else "inverse",
        )
    with col2:
        model_name = "Random Forest" if health["connected"] else "—"
        st.metric(
            label="Inference Pipeline",
            value=model_name,
            help="Trained Random Forest pipeline encapsulated in FastAPI service.",
        )
    with col3:
        st.metric(
            label="Monitored Sensors",
            value="6 Channels",
            help="Approved features: Type, Air Temp, Process Temp, Speed, Torque, Tool Wear.",
        )
    with col4:
        st.metric(
            label="Decision Cutoff",
            value="45.0%",
            help="Tuned threshold (0.45) selected via 5-Fold Stratified Cross-Validation in Phase 3.5.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Overview Container
    with st.container():
        st.subheader("Decoupled System Architecture")
        st.markdown(
            f"""
            FactoryPulse adopts a decoupled, microservice-ready architecture separating presentation from machine learning inference:
            
            - **Web Dashboard**: Streamlit interface running on local port `8501`. Collects operator input, 
              validates constraints, and renders predictions.
            - **Backend Prediction Service**: FastAPI REST API running on `{client.base_url}`. 
              Serves `/health`, `/model/info`, and `/predict`.
            - **ML Model Pipeline**: Balanced Random Forest with embedded `ColumnTransformer` preprocessing. 
              No model weights are loaded directly into the dashboard.
            - **Decision Rule**: Explicit classification cutoff at **45.0%** failure probability 
              (`probability >= 0.45`).
            """
        )

        if not health["connected"]:
            st.warning(f"⚠️ **Backend Service Notice**: {health['message']}")
        else:
            st.success("✅ **Integration Active**: Dashboard is successfully communicating with the FastAPI prediction service.")


# =============================================================================
# 2. Sensor Monitoring Section
# =============================================================================

def render_sensor_monitoring_section():
    """Render physical sensor channel surveillance and parameter documentation."""
    st.header("📡 Sensor Monitoring")
    st.write(
        "Real-time sensor telemetry surveillance across physical thermal, rotational, and wear metrics."
    )

    # Sensor Channels Grid
    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container():
            st.markdown("#### 🌡️ Thermal Subsystem")
            st.metric(label="Air Temperature [K]", value="Ambient Feed", help="Sensors capture ambient air temperature [K].")
            st.metric(label="Process Temperature [K]", value="Operating Heat", help="Process temperature at cutting interface [K].")
            st.caption("Operating Range: ~295 K to 305 K.")

    with col2:
        with st.container():
            st.markdown("#### ⚡ Spindle Dynamics")
            st.metric(label="Rotational Speed [rpm]", value="Motor Velocity", help="Spindle motor velocity [RPM].")
            st.metric(label="Torque [Nm]", value="Motor Resistance", help="Torque load resistance [Nm].")
            st.caption("Inverse physical relationship between speed and torque.")

    with col3:
        with st.container():
            st.markdown("#### ⏱️ Tool Cutter State")
            st.metric(label="Tool Wear [min]", value="Cutter Age", help="Cumulative contact cutting minutes.")
            st.metric(label="Machine Variant", value="L / M / H", help="Quality variant workload rating.")
            st.caption("Wear accelerates thermal and torque stresses.")

    st.markdown("---")

    # Operating Parameter Reference Guide
    st.subheader("Approved Operating Parameter Reference Guide")
    st.markdown(
        """
        | Sensor Parameter | Schema Key | Symbol / Unit | Physical Boundary | Data Type |
        | :--- | :--- | :--- | :--- | :--- |
        | **Machine Variant** | `type` | Variant (L / M / H) | `["L", "M", "H"]` | String |
        | **Air Temperature** | `air_temperature_k` | Kelvin [K] | `> 0.0 K` | Float |
        | **Process Temperature** | `process_temperature_k` | Kelvin [K] | `> 0.0 K` | Float |
        | **Rotational Speed** | `rotational_speed_rpm` | RPM [rpm] | `> 0 rpm` | Integer |
        | **Torque** | `torque_nm` | Newton-meters [Nm] | `≥ 0.0 Nm` | Float |
        | **Tool Wear** | `tool_wear_min` | Minutes [min] | `≥ 0 min` | Integer |
        """
    )


# =============================================================================
# 3. Failure Prediction Section (Phase 5.2 Core)
# =============================================================================

def render_failure_prediction_section():
    """Render the interactive prediction form, submit telemetry, and display API results."""
    st.header("🎯 Failure Prediction & Risk Assessment")
    st.write(
        "Submit physical machine telemetry to the FactoryPulse FastAPI service to evaluate breakdown risk "
        "against the calibrated 45.0% decision cutoff."
    )

    client = get_api_client()
    health = check_cached_health()

    # Connection Status Alert Banner
    if not health["connected"]:
        st.error(
            f"🔴 **API Server Disconnected**: Cannot reach FastAPI at `{client.base_url}`. "
            "Please start the backend server before submitting prediction requests:\n\n"
            "```powershell\npython src/api/run_api.py\n```"
        )
    else:
        st.caption(f"🟢 Connected to `{client.base_url}/predict`")

    # Initialize Session State
    if "latest_prediction" not in st.session_state:
        st.session_state.latest_prediction = None
    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None

    # Telemetry Input Form
    with st.form("machine_prediction_form"):
        st.subheader("Machine Telemetry Input")
        st.write("Enter operating sensor values. All 6 physical parameters are required:")

        col1, col2, col3 = st.columns(3)

        with col1:
            machine_id = st.text_input(
                "Machine Identifier",
                value="CNC-UNIT-01",
                help="Optional machine tag for tracking (e.g., CNC-01).",
            )
            type_option = st.selectbox(
                "Machine Variant (Type)",
                options=["L", "M", "H"],
                index=1,
                help="Product variant classification: 'L' (Low), 'M' (Medium), 'H' (High).",
            )

        with col2:
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=200.0,
                max_value=400.0,
                value=298.5,
                step=0.1,
                format="%.1f",
                help="Ambient air temperature in Kelvin (must be > 0).",
            )
            process_temp = st.number_input(
                "Process Temperature [K]",
                min_value=200.0,
                max_value=450.0,
                value=308.7,
                step=0.1,
                format="%.1f",
                help="Process cutting temperature in Kelvin (must be > 0).",
            )

        with col3:
            rot_speed = st.number_input(
                "Rotational Speed [rpm]",
                min_value=1,
                max_value=5000,
                value=1500,
                step=10,
                help="Spindle motor velocity in revolutions per minute (must be > 0).",
            )
            torque = st.number_input(
                "Torque [Nm]",
                min_value=0.0,
                max_value=250.0,
                value=45.0,
                step=0.5,
                format="%.1f",
                help="Spindle torque in Newton-meters (must be ≥ 0).",
            )

        tool_wear = st.number_input(
            "Cumulative Tool Wear [min]",
            min_value=0,
            max_value=500,
            value=100,
            step=1,
            help="Cumulative tool cutter contact minutes (must be ≥ 0).",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_prediction = st.form_submit_button(
            "🚀 Predict Machine Health",
            type="primary",
            use_container_width=True,
        )

    # Process Form Submission
    if submit_prediction:
        payload = {
            "machine_id": machine_id.strip() if machine_id.strip() else None,
            "type": type_option,
            "air_temperature_k": float(air_temp),
            "process_temperature_k": float(process_temp),
            "rotational_speed_rpm": int(rot_speed),
            "torque_nm": float(torque),
            "tool_wear_min": int(tool_wear),
        }

        with st.spinner("Submitting telemetry to FactoryPulse API..."):
            api_resp = client.predict(payload)

        if api_resp.success and api_resp.data:
            # Preserve successful prediction in session state
            st.session_state.latest_prediction = api_resp.data
            st.session_state.prediction_error = None
            st.success("✅ Prediction generated successfully!")
        else:
            # Capture error without wiping previous valid prediction
            st.session_state.prediction_error = api_resp.error_message
            st.error(f"❌ {api_resp.error_message}")

    # Display Current Error if Present
    if st.session_state.prediction_error and not submit_prediction:
        st.error(f"⚠️ **Last Request Error**: {st.session_state.prediction_error}")

    st.markdown("---")

    # Display Actual Prediction Results from Session State
    st.subheader("📊 Actual Prediction Results")

    latest = st.session_state.latest_prediction

    if latest is None:
        st.info("ℹ️ No prediction has been executed yet. Configure the sensor values above and click **Predict Machine Health**.")
        
        # Standby metric placeholders
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Failure Probability", "— %")
        with res_col2:
            st.metric("Applied Threshold", "45.0 %")
        with res_col3:
            st.metric("Predicted Status", "Awaiting Evaluation")
    else:
        # Extract actual values returned by the FastAPI prediction service
        prob_pct = latest["failure_probability"] * 100.0
        thresh_pct = latest["applied_threshold"] * 100.0
        pred_class = latest["predicted_class"]
        pred_label = latest["prediction_label"]
        returned_mid = latest.get("machine_id") or "Unspecified"
        model_name = latest.get("model_name", "RandomForestClassifier (balanced)")

        # Main Metric Row
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Failure Probability",
                value=f"{prob_pct:.1f} %",
                help="Estimated failure probability produced by the Random Forest model pipeline.",
            )
        with res_col2:
            st.metric(
                label="Decision Cutoff",
                value=f"{thresh_pct:.1f} %",
                help="Calibrated threshold: Machine Failure if Probability ≥ 45.0%.",
            )
        with res_col3:
            status_text = "FAILURE DETECTED" if pred_class == 1 else "NORMAL"
            st.metric(
                label="Binary Outcome",
                value=f"Class {pred_class} ({status_text})",
                help="Predicted class 0 (Normal) or class 1 (Machine Failure).",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Detailed Status Card with clear threshold comparison
        with st.container():
            if pred_class == 1:
                st.error(
                    f"### 🚨 Classification: Machine Failure Risk Detected\n\n"
                    f"- **Machine Identifier**: `{returned_mid}`\n"
                    f"- **Estimated Probability**: **{prob_pct:.2f}%** (Exceeds decision threshold of **{thresh_pct:.1f}%**)\n"
                    f"- **API Classification**: `{pred_label}` (Class 1)\n"
                    f"- **Inference Model**: `{model_name}`\n\n"
                    f"**Action Recommended**: Schedule preventive maintenance inspection for machine `{returned_mid}`."
                )
            else:
                st.success(
                    f"### ✅ Classification: Normal Machine Operation\n\n"
                    f"- **Machine Identifier**: `{returned_mid}`\n"
                    f"- **Estimated Probability**: **{prob_pct:.2f}%** (Safely below decision threshold of **{thresh_pct:.1f}%**)\n"
                    f"- **API Classification**: `{pred_label}` (Class 0)\n"
                    f"- **Inference Model**: `{model_name}`\n\n"
                    f"**Operating Status**: Machine `{returned_mid}` is operating within safe physical parameters."
                )

        # Clear Note on Risk Tiers (Mandatory compliance)
        st.caption(
            "📌 **Engineering Note**: In strict accordance with project standards, arbitrary risk tiers "
            "(LOW, MEDIUM, HIGH, CRITICAL) are omitted. Classification strictly reflects the calibrated "
            "decision rule: `failure_probability >= 0.45`."
        )


# =============================================================================
# Main Application Flow
# =============================================================================

def main():
    """Main dashboard entry point."""
    # Render common header
    render_header()

    # Render sidebar and capture navigation choice
    selected_page = render_sidebar()

    # Route to selected section
    if selected_page == "Overview":
        render_overview_section()
    elif selected_page == "Machine Monitoring":
        render_sensor_monitoring_section()
    elif selected_page == "Failure Prediction":
        render_failure_prediction_section()

    # Bottom footer
    st.markdown("---")
    st.caption(
        f"FactoryPulse v0.5.2-integrated | Decoupled FastAPI Client | {DISCLAIMER_TEXT}"
    )


if __name__ == "__main__":
    main()
