import unittest

import pandas as pd

from src.io_climate.viz import (
    compute_relative_deviation,
    plot_relative_deviation_map,
)


class RelativeDeviationMapTests(unittest.TestCase):
    def test_compute_relative_deviation_positive_negative_and_zero(self):
        df = pd.DataFrame(
            {
                "country": ["IT", "FR", "DE"],
                "loss_pct": [12.0, 10.0, 8.0],
            }
        )

        out = compute_relative_deviation(df)

        self.assertAlmostEqual(out.loc[0, "loss_pct_deviation"], 2.0)
        self.assertAlmostEqual(out.loc[1, "loss_pct_deviation"], 0.0)
        self.assertAlmostEqual(out.loc[2, "loss_pct_deviation"], -2.0)

    def test_extreme_variance_uses_symmetric_divergent_scale(self):
        df = pd.DataFrame(
            {
                "country": ["IT", "FR", "DE"],
                "loss_pct": [600.0, 100.0, 99.5],
                "loss_abs": [1200.0, 75.0, 12.0],
            }
        )
        fig = plot_relative_deviation_map(df)

        choropleth = next(trace for trace in fig.data if trace.type == "choropleth")

        self.assertAlmostEqual(float(choropleth.zmid), 0.0)
        self.assertAlmostEqual(abs(float(choropleth.zmin)), abs(float(choropleth.zmax)))
        self.assertEqual(choropleth.colorscale[0][1], "rgb(5,48,97)")
        self.assertEqual(choropleth.colorbar.title.text, "Loss % deviation")
        self.assertEqual(choropleth.colorbar.tickformat, ".1e")
        self.assertIn("Loss impact (%)", choropleth.hovertemplate)
        self.assertIn("Loss impact (abs)", choropleth.hovertemplate)

    def test_missing_baseline_metric_fails_gracefully(self):
        df = pd.DataFrame({"country": ["IT", "FR"], "wrong_metric": [1.0, 2.0]})

        with self.assertRaisesRegex(ValueError, "Baseline metric 'loss_pct' is missing"):
            compute_relative_deviation(df)


if __name__ == "__main__":
    unittest.main()
