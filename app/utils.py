import os
import sys
from pathlib import Path
from typing import Optional
import logging

# Configure logger
logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent.resolve()


def setup_paths():
    """Ensures src/ is importable."""
    root = get_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))


def get_spark_session(app_name: str = "StreamlitHazardApp"):
    """
    Returns a Spark Session.
    Prioritizes DatabricksSession, handles [CONNECT_URL_NOT_SET] by forcing local mode.
    """

    # --- 1. Databricks Runtime (Cluster/App) ---
    if os.environ.get("DATABRICKS_RUNTIME_VERSION"):
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()

    # --- 2. Local Development (Strategy A: Databricks Connect) ---
    try:
        from databricks.connect import DatabricksSession

        profile = os.environ.get("DATABRICKS_CONFIG_PROFILE", "local")

        # Note: .appName() might not be available on all DatabricksSession builders
        # We focus on getting the connection first using the profile + serverless
        builder = DatabricksSession.builder.profile(profile).serverless()

        return builder.getOrCreate()

    except Exception as e:
        logger.warning(f"Strategy A (DatabricksSession) failed: {e}")

    # --- 3. Final Fallback (Strategy B: Truly Local Spark, No Connect) ---
    try:
        from pyspark.sql import SparkSession

        logger.info("Strategy B: Initializing truly Local Spark (non-connect).")

        # We explicitly set .master("local[*]") to prevent Connect from hijacking
        # and trying to reach a remote URL that isn't set.
        return (
            SparkSession.builder.master("local[*]")
            .appName(app_name)
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
            .getOrCreate()
        )
    except Exception as e2:
        logger.error(f"Strategy B failed: {e2}")
        raise e2
