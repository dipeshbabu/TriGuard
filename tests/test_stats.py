import sys
import tempfile
import unittest

import pandas as pd


_REAL_VERSION_INFO = sys.version_info
sys.version_info = (3, 10, 12, "final", 0)
try:
    from triguard.stats import (
        mann_whitney_grid,
        paired_design_sensitivity,
        paired_regularizer_tests,
        primary_paired_tests,
        summarize_with_ci,
    )
finally:
    sys.version_info = _REAL_VERSION_INFO


class StatisticsTests(unittest.TestCase):
    def test_primary_outputs_keep_headers_when_no_primary_rows_are_eligible(self):
        frame = pd.DataFrame(
            [
                {
                    "dataset": "cifar10",
                    "model": "resnet50_imagenet",
                    "seed": 0,
                    "faithfulness_enabled": 0,
                    "lambda_entropy": 0.0,
                    "lambda_wads": 0.05,
                    "lambda_rar": 0.0,
                    "lambda_far": 0.0,
                    "lambda_curvature": 0.0,
                    "lambda_robust": 0.0,
                    "lambda_attr_mass": 0.0,
                    "reference_risk": "max",
                    "heldout_wads_mean": 0.5,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as out_dir:
            result = primary_paired_tests(frame, out_dir)
            tests_csv = pd.read_csv(f"{out_dir}/primary_paired_tests.csv")
            decisions_csv = pd.read_csv(f"{out_dir}/primary_decisions.csv")
        self.assertTrue(result.empty)
        self.assertIn("p_value_holm", tests_csv.columns)
        self.assertIn("success", decisions_csv.columns)

    def test_architecture_tests_do_not_group_by_model_specific_condition_hash(self):
        rows = []
        for seed in range(3):
            rows.extend(
                [
                    {
                        "dataset": "cifar10",
                        "model": "resnet50",
                        "seed": seed,
                        "condition_hash": "resnet-condition",
                        "lambda_entropy": 0.0,
                        "clean_acc": 0.8 + seed * 0.01,
                    },
                    {
                        "dataset": "cifar10",
                        "model": "densenet121",
                        "seed": seed,
                        "condition_hash": "densenet-condition",
                        "lambda_entropy": 0.0,
                        "clean_acc": 0.7 + seed * 0.01,
                    },
                ]
            )
        with tempfile.TemporaryDirectory() as out_dir:
            result = mann_whitney_grid(pd.DataFrame(rows), out_dir)
        accuracy_rows = result[result["metric"] == "clean_acc"]
        self.assertEqual(len(accuracy_rows), 1)

    def test_design_sensitivity_is_written_as_planning_approximation(self):
        with tempfile.TemporaryDirectory() as out_dir:
            result = paired_design_sensitivity(out_dir, paired_n=10)
        self.assertEqual(result.iloc[0]["approximation"], "two_sided_paired_t")
        self.assertGreater(float(result.iloc[0]["standardized_paired_effect"]), 0.0)

    def test_primary_contrasts_are_explicit_and_seed_paired(self):
        rows = []
        settings = [
            ("ce", 0.0, 0.0, 0.0, 0.0, "max"),
            ("mean", 0.05, 0.0, 0.0, 0.0, "mean"),
            ("max_only", 0.05, 0.0, 0.0, 0.0, "max"),
            ("max_controls", 0.05, 0.01, 0.25, 0.0, "max"),
            ("cvar", 0.05, 0.01, 0.25, 0.01, "cvar"),
        ]
        for seed in range(10):
            for name, wads, curvature, robust, mass, risk in settings:
                rows.append(
                    {
                        "dataset": "cifar10",
                        "model": "resnet50_imagenet",
                        "seed": seed,
                        "lambda_entropy": 0.0,
                        "lambda_wads": wads,
                        "lambda_rar": 0.0,
                        "lambda_far": 0.0,
                        "lambda_curvature": curvature,
                        "lambda_robust": robust,
                        "lambda_attr_mass": mass,
                        "reference_risk": risk,
                        "heldout_wads_mean": (
                            1.0
                            - {
                                "ce": 0.0,
                                "mean": 0.1,
                                "max_only": 0.2,
                                "max_controls": 0.2,
                                "cvar": 0.3,
                            }[name]
                            + seed * 0.001
                        ),
                        "clean_acc": 0.8,
                        "adv_error": 0.2,
                        "heldout_attr_mass_ratio_below_floor_rate": 0.0,
                        "ig_del_auc_mean": 0.2,
                        "ig_ins_auc_mean": 0.8,
                    }
                )
        with tempfile.TemporaryDirectory() as out_dir:
            result = primary_paired_tests(pd.DataFrame(rows), out_dir)
            decisions = pd.read_csv(f"{out_dir}/primary_decisions.csv")
        metric_rows = result[result["metric"] == "heldout_wads_mean"]
        self.assertEqual(len(metric_rows), 2)
        self.assertTrue((metric_rows["paired_n"] == 10).all())
        self.assertEqual(
            set(zip(metric_rows["treatment_a"], metric_rows["treatment_b"])),
            {
                ("mean_reference_only", "max_reference_only"),
                ("max_reference_controls", "cvar_mass"),
            },
        )
        self.assertIn("p_value_holm", metric_rows.columns)
        self.assertTrue(decisions["success"].all())

    def test_summary_does_not_pool_entropy_configurations(self):
        frame = pd.DataFrame(
            [
                {
                    "dataset": "mnist",
                    "model": "simplecnn",
                    "lambda_entropy": 0.0,
                    "lambda_wads": 0.0,
                    "clean_acc": 0.8,
                },
                {
                    "dataset": "mnist",
                    "model": "simplecnn",
                    "lambda_entropy": 0.05,
                    "lambda_wads": 0.0,
                    "clean_acc": 0.9,
                },
            ]
        )
        with tempfile.TemporaryDirectory() as out_dir:
            result = summarize_with_ci(frame, out_dir)
        accuracy_rows = result[result["metric"] == "clean_acc"]
        self.assertEqual(len(accuracy_rows), 2)
        self.assertEqual(set(accuracy_rows["lambda_entropy"]), {0.0, 0.05})

    def test_regularizer_comparison_pairs_matching_seeds(self):
        rows = []
        for seed in range(3):
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": "resnet50_imagenet",
                    "input_profile": "imagenet",
                    "seed": seed,
                    "lambda_entropy": 0.0,
                    "lambda_wads": 0.0,
                    "lambda_rar": 0.0,
                    "lambda_far": 0.0,
                    "lambda_curvature": 0.0,
                    "lambda_robust": 0.0,
                    "wads_mean": 1.0 + seed * 0.1,
                }
            )
            rows.append(
                {
                    "dataset": "cifar10",
                    "model": "resnet50_imagenet",
                    "input_profile": "imagenet",
                    "seed": seed,
                    "lambda_entropy": 0.0,
                    "lambda_wads": 0.05,
                    "lambda_rar": 0.0,
                    "lambda_far": 0.0,
                    "lambda_curvature": 0.0,
                    "lambda_robust": 0.0,
                    "wads_mean": 0.5 + seed * 0.1,
                }
            )

        with tempfile.TemporaryDirectory() as out_dir:
            result = paired_regularizer_tests(pd.DataFrame(rows), out_dir)

        metric_rows = result[result["metric"] == "wads_mean"]
        self.assertEqual(len(metric_rows), 1)
        self.assertEqual(int(metric_rows.iloc[0]["paired_n"]), 3)
        self.assertAlmostEqual(
            float(metric_rows.iloc[0]["mean_delta_b_minus_a"]), -0.5
        )
        self.assertIn("p_value_holm", metric_rows.columns)


if __name__ == "__main__":
    unittest.main()
