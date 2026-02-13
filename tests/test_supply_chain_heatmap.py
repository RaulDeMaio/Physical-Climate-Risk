import unittest

import pandas as pd

from src.io_climate.supply_chain_heatmap import (
    build_heatmap_frame,
    map_coordinates_to_viewport,
    normalize_intensity,
)


class SupplyChainHeatmapTests(unittest.TestCase):
    def test_normalize_intensity_quantile_capping(self):
        values = pd.Series([0.0, 2.0, 4.0, 100.0])
        norm = normalize_intensity(values, quantile_cap=0.75)

        self.assertEqual(float(norm.iloc[0]), 0.0)
        self.assertAlmostEqual(float(norm.iloc[1]), 0.5, places=6)
        self.assertAlmostEqual(float(norm.iloc[2]), 1.0, places=6)
        self.assertAlmostEqual(float(norm.iloc[3]), 1.0, places=6)

    def test_build_heatmap_frame_empty_and_null(self):
        empty = pd.DataFrame(columns=["source", "target", "delta", "delta_rel"])
        empty_matrix, empty_norm, empty_clip = build_heatmap_frame(
            empty, perspective="absolute"
        )
        self.assertTrue(empty_matrix.empty)
        self.assertTrue(empty_norm.empty)
        self.assertEqual(empty_clip, 0.0)

        null_df = pd.DataFrame(
            {
                "source": ["A", "A"],
                "target": ["B", "C"],
                "delta": [None, None],
                "delta_rel": [None, None],
            }
        )
        null_matrix, null_norm, null_clip = build_heatmap_frame(
            null_df, perspective="percentage"
        )
        self.assertEqual(float(null_matrix.fillna(0).to_numpy().sum()), 0.0)
        self.assertEqual(float(null_norm.fillna(0).to_numpy().sum()), 0.0)
        self.assertEqual(null_clip, 0.0)

    def test_build_heatmap_aggregates_by_sector(self):
        links = pd.DataFrame(
            {
                "i_label": ["IT::A", "IT::A", "FR::B"],
                "j_label": ["FR::B", "DE::B", "IT::A"],
                "delta": [1.0, 2.0, 3.0],
                "delta_rel": [0.1, 0.2, 0.3],
            }
        )
        matrix, _, _ = build_heatmap_frame(
            links,
            perspective="absolute",
            aggregation="sector",
        )

        self.assertEqual(matrix.loc["A", "B"], 3.0)
        self.assertEqual(matrix.loc["B", "A"], 3.0)

    def test_coordinate_bounds_clip(self):
        points = pd.DataFrame(
            {
                "x": [-10, 50, 110],
                "y": [-5, 50, 200],
            }
        )
        mapped = map_coordinates_to_viewport(
            points,
            x_col="x",
            y_col="y",
            bounds=(0, 100, 0, 100),
            mode="clip",
        )

        self.assertEqual(mapped.loc[0, "x_mapped"], 0)
        self.assertEqual(mapped.loc[0, "y_mapped"], 0)
        self.assertEqual(mapped.loc[2, "x_mapped"], 100)
        self.assertEqual(mapped.loc[2, "y_mapped"], 100)
        self.assertEqual(int(mapped["in_bounds"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
