# src/data_io/sector_decoder.py

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_NACE_MAPPING_TABLE = "openeconomics_prod.gold.dim_nace_sector_mapping"


def build_sector_decoder(
    spark: SparkSession,
    sectors_in_sam: Optional[Sequence[str]] = None,
    table_name: str = DEFAULT_NACE_MAPPING_TABLE,
) -> Dict[str, str]:
    """
    Build a dictionary mapping SAM sector codes (e.g. 'P_C10-12', 'P_L')
    to human-readable names using openeconomics_prod.gold.dim_nace_sector_mapping.

    Rules (priority):
      1) Prefer 'P_' + estat_s63g_code  -> estat_s63g_des
      2) Fallback 'P_' + estat_s11g_code -> estat_s11g_des
      3) If not found, caller can fallback to the original sector code.

    Parameters
    ----------
    spark
        Active SparkSession / Spark Connect session.
    sectors_in_sam
        Optional list of SAM sector codes present in the current run (e.g. ['P_C10-12','P_L']).
        If provided, the mapping table is filtered to this subset to reduce data transfer.
    table_name
        Fully qualified mapping table name.

    Returns
    -------
    sector_decoder : dict
        {'P_C10-12': 'Manufacture of food products; ...', ...}
    """
    nace_df = spark.table(table_name)

    # Helper to standardize codes (trim, drop empties)
    def _valid_code(colname: str):
        return F.col(colname).isNotNull() & (F.length(F.trim(F.col(colname))) > 0)

    # Preferred mapping: s63g
    map_s63 = (
        nace_df
        .filter(_valid_code("estat_s63g_code"))
        .select(
            F.concat(F.lit("P_"), F.trim(F.col("estat_s63g_code"))).alias("sector"),
            F.trim(F.col("estat_s63g_des")).alias("sector_name"),
        )
        .dropDuplicates(["sector"])
    )

    # Fallback mapping: s11g
    map_s11 = (
        nace_df
        .filter(_valid_code("estat_s11g_code"))
        .select(
            F.concat(F.lit("P_"), F.trim(F.col("estat_s11g_code"))).alias("sector"),
            F.trim(F.col("estat_s11g_des")).alias("sector_name"),
        )
        .dropDuplicates(["sector"])
    )

    # Priority rule: keep s11 only if not already mapped by s63
    decoder_df = map_s63.unionByName(
        map_s11.join(map_s63.select("sector"), on="sector", how="left_anti")
    )

    # Optional: restrict to sectors appearing in SAM to minimize Spark->Python transfer
    if sectors_in_sam is not None:
        wanted = list(dict.fromkeys(sectors_in_sam))  # stable unique
        decoder_df = decoder_df.filter(F.col("sector").isin(wanted))

    # Collect small two-column mapping
    rows = decoder_df.select("sector", "sector_name").collect()
    return {r["sector"]: r["sector_name"] for r in rows}
