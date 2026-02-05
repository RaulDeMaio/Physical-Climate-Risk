from dataclasses import dataclass
from typing import Dict, List, Optional
import logging
import traceback

import numpy as np
import pandas as pd
import streamlit as st

# Internal Imports
from app.utils import get_spark_session
from data_io.eurostat_sam import load_sam_latest_year, extract_model_inputs_from_sam
from src.data_io.sector_decoder import build_sector_decoder
from src.io_climate.calibration import (
    CalibrationPaths,
    load_losses_from_excel,
    build_intensity_panel,
    build_percentile_table,
)
from src.data_io.eurostat_output import (
    EurostatOutputConfig,
    fetch_output_2024_prices_mn_eur,
)
from io_climate.model import IOClimateModel

# --- Constants ---
LOSS_XLSX_PATH = "src/data_io/input_data.xlsx"
API_TXT_PATH = "src/data_io/Output_API.txt"
CACHE_DIR = "data/cache"

logger = logging.getLogger(__name__)


@dataclass
class ModelInputs:
    """Container for immutable model inputs."""

    Z: np.ndarray
    FD: np.ndarray
    X: np.ndarray
    A: np.ndarray
    globsec_of: np.ndarray
    node_labels: List[str]
    sector_decoder: Dict[str, str]
    latest_year: int
    model_instance: IOClimateModel
    is_mock: bool = False
    error_msg: Optional[str] = None
    stack_trace: Optional[str] = None


def create_mock_data(
    error: Optional[str] = None, stack: Optional[str] = None
) -> ModelInputs:
    """Creates synthetic data for testing UI when DB is unreachable."""
    st.warning("⚠️ Running in Mock Mode: Database Unreachable. Data is randomized.")
    if error:
        with st.expander("🔍 Connectivity Error Details"):
            st.error(error)
            if stack:
                st.code(stack, language="python")

    n_countries = 3
    n_sectors = 5
    countries = ["IT", "DE", "FR"]
    sectors = [f"P_S{i}" for i in range(n_sectors)]

    node_labels = [f"{c}::{s}" for c in countries for s in sectors]
    n = len(node_labels)

    np.random.seed(42)
    Z = np.random.rand(n, n) * 100
    FD = np.random.rand(n) * 1000
    X = Z.sum(axis=1) + FD
    A = Z / X
    A = np.nan_to_num(A)

    globsec_of = np.tile(np.arange(n_sectors), n_countries)
    decoder = {s: f"Sector {s}" for s in sectors}

    model = IOClimateModel(
        Z=Z, FD=FD, X=X, globsec_of=globsec_of, A=A, node_labels=node_labels
    )

    return ModelInputs(
        Z=Z,
        FD=FD,
        X=X,
        A=A,
        globsec_of=globsec_of,
        node_labels=node_labels,
        sector_decoder=decoder,
        latest_year=2024,
        model_instance=model,
        is_mock=True,
        error_msg=error,
        stack_trace=stack,
    )


@st.cache_resource(show_spinner="Connecting to Data Source...")
def load_core_model_data() -> ModelInputs:
    """
    Tries to load SAM data. Catching ALL exceptions including runtime connectivity issues.
    """
    try:
        # 1. Initialize Spark (Strategy A then B)
        spark = get_spark_session()

        # 2. Load SAM (This often triggers the actual lazy connection check)
        sam_df, year = load_sam_latest_year(spark)

        # 3. Extract Matrices
        Z, FD, X, A, globsec_of, node_labels = extract_model_inputs_from_sam(sam_df)

        # 4. Build Sector Decoder
        sectors_in_sam = sorted({lbl.split("::")[1] for lbl in node_labels})
        decoder = build_sector_decoder(spark, sectors_in_sam=sectors_in_sam)

        # 5. Instantiate Base Model
        model = IOClimateModel(
            Z=Z, FD=FD, X=X, globsec_of=globsec_of, A=A, node_labels=node_labels
        )

        return ModelInputs(
            Z=Z,
            FD=FD,
            X=X,
            A=A,
            globsec_of=globsec_of,
            node_labels=node_labels,
            sector_decoder=decoder,
            latest_year=year,
            model_instance=model,
            is_mock=False,
        )

    except Exception as e:
        err_str = str(e)
        stack = traceback.format_exc()
        logger.error(f"Data loading failed: {err_str}")
        return create_mock_data(error=err_str, stack=stack)


@st.cache_resource(show_spinner="Loading Calibration Data...")
def load_calibration_data() -> pd.DataFrame:
    try:
        import pathlib

        pathlib.Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        paths = CalibrationPaths(losses_xlsx_path=LOSS_XLSX_PATH)
        df_type, df_year = load_losses_from_excel(paths)
        api_url = pathlib.Path(API_TXT_PATH).read_text().strip()
        cache_path = pathlib.Path(CACHE_DIR) / "eurostat_p1_output_2024.parquet"
        out_cfg = EurostatOutputConfig(api_url=api_url, cache_path=str(cache_path))
        df_output = fetch_output_2024_prices_mn_eur(config=out_cfg)
        intensity_panel = build_intensity_panel(
            df_type=df_type, df_year=df_year, df_output=df_output, geo_col_output="geo"
        )
        return build_percentile_table(intensity_panel)
    except Exception as e:
        logger.error(f"Calibration load failed: {e}")
        return pd.DataFrame(
            {
                "ISO2": ["IT", "DE", "FR"] * 4,
                "hazard": ["meteorological"] * 3 + ["hydrological"] * 9,
                "extreme": [0.05, 0.02, 0.03] * 4,
                "severe": [0.03, 0.01, 0.02] * 4,
                "moderate": [0.01, 0.005, 0.01] * 4,
                "very_extreme": [0.10, 0.05, 0.06] * 4,
            }
        )
