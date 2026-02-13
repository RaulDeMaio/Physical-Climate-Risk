import time
import unittest

import numpy as np
import pandas as pd

from src.io_climate.viz import (
    compute_relative_deviation,
    plot_supply_chain_deviation_bars,
    prepare_supply_chain_deviation_frame,
)


class SupplyChainDeviationPlotTests(unittest.TestCase):
    def test_deviation_math_matches_mean_centered_definition(self):
        df = pd.DataFrame(
            {
                "sector": ["A", "B", "C"],
                "loss_pct": [10.0, 8.0, 6.0],
                "loss_abs": [100.0, 80.0, 60.0],
            }
        )

        out = compute_relative_deviation(
            df,
            metric="loss_pct",
            output_col="loss_pct_deviation",
        )

        self.assertAlmostEqual(float(out.loc[0, "loss_pct_deviation"]), 2.0)
        self.assertAlmostEqual(float(out.loc[1, "loss_pct_deviation"]), 0.0)
        self.assertAlmostEqual(float(out.loc[2, "loss_pct_deviation"]), -2.0)

    def test_boundary_zero_deviation_and_missing_data(self):
        df = pd.DataFrame(
            {
                "sector": ["A", "B", "C"],
                "loss_pct": [10.0, np.nan, 10.0],
                "loss_abs": [100.0, 95.0, 90.0],
            }
        )

        out = prepare_supply_chain_deviation_frame(
            df,
            baseline_metric="loss_pct",
            deviation_metric="loss_pct_deviation",
            ranking_mode="full_distribution",
        )

        row_a = out.loc[out["sector"] == "A", "loss_pct_deviation"].iloc[0]
        row_c = out.loc[out["sector"] == "C", "loss_pct_deviation"].iloc[0]
        row_b = out.loc[out["sector"] == "B", "loss_pct_deviation"].iloc[0]

        self.assertAlmostEqual(float(row_a), 0.0)
        self.assertAlmostEqual(float(row_c), 0.0)
        self.assertTrue(np.isnan(float(row_b)))

        fig = plot_supply_chain_deviation_bars(df, ranking_mode="full_distribution")
        self.assertIn("Deviation from mean relative sectoral impact", fig.layout.xaxis.title.text)

    def test_top_bottom_ranking_cap(self):
        df = pd.DataFrame(
            {
                "sector": [f"S{i}" for i in range(80)],
                "loss_pct": np.linspace(-40.0, 39.0, 80),
                "loss_abs": np.linspace(100.0, 1000.0, 80),
            }
        )

        out = prepare_supply_chain_deviation_frame(
            df,
            baseline_metric="loss_pct",
            deviation_metric="loss_pct_deviation",
            ranking_mode="top_bottom",
            top_k=20,
        )
        self.assertLessEqual(len(out), 40)

    def test_large_input_performance(self):
        n_points = 10_000
        df = pd.DataFrame(
            {
                "sector": [f"S{i}" for i in range(n_points)],
                "loss_pct": np.sin(np.linspace(0, 120, n_points)) * 10.0,
                "loss_abs": np.linspace(1.0, 25000.0, n_points),
            }
        )

        start = time.perf_counter()
        out = prepare_supply_chain_deviation_frame(
            df,
            baseline_metric="loss_pct",
            deviation_metric="loss_pct_deviation",
            ranking_mode="top_bottom",
            top_k=20,
        )
        _ = plot_supply_chain_deviation_bars(out, ranking_mode="full_distribution")
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
