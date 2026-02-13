import unittest

import pandas as pd

from src.io_climate.viz import plot_combined_linkage_deviation


class CombinedLinkageDeviationTests(unittest.TestCase):
    def test_combines_strengthened_and_weakened_into_single_diverging_plot(self):
        weakened = pd.DataFrame(
            {
                "i_label": ["IT::A", "FR::B"],
                "j_label": ["DE::C", "ES::D"],
                "delta": [-0.005, -0.002],
                "delta_rel": [-0.2, -0.1],
            }
        )
        strengthened = pd.DataFrame(
            {
                "i_label": ["NL::E", "BE::F"],
                "j_label": ["IT::G", "DE::H"],
                "delta": [0.003, 0.001],
                "delta_rel": [0.15, 0.05],
            }
        )

        fig = plot_combined_linkage_deviation(strengthened, weakened, top_k=2)

        self.assertGreaterEqual(len(fig.data), 1)
        self.assertEqual(fig.layout.xaxis.title.text, "ΔA (percentage points)")
        self.assertTrue(any(getattr(shape, "x0", None) == 0.0 for shape in (fig.layout.shapes or [])))

    def test_handles_empty_inputs(self):
        empty = pd.DataFrame(columns=["i_label", "j_label", "delta", "delta_rel"])
        fig = plot_combined_linkage_deviation(empty, empty)
        self.assertEqual(fig.layout.xaxis.visible, False)


if __name__ == "__main__":
    unittest.main()
