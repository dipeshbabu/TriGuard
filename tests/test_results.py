import csv
import hashlib
import os
import sys
import tempfile
import unittest
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import torch


_REAL_VERSION_INFO = sys.version_info
sys.version_info = (3, 10, 12, "final", 0)
try:
    from run_triguard import (
        experiment_identity,
        get_experiment_grid,
        parse_seeds,
        checkpoint_artifacts_complete,
        resolve_checkpoint_path,
        train_model,
        write_checkpoint_metadata,
    )
    from triguard.build_references import verify_calibration_checkpoint
    from triguard.protocol import validate_protocol_values
    from triguard.results import append_csv, csv_contains_identity
finally:
    sys.version_info = _REAL_VERSION_INFO


def _write_result_in_process(payload):
    path, index = payload
    append_csv(
        path,
        {"config_hash": str(index), "metric": float(index)},
        ["config_hash", "metric"],
        key_fields=["config_hash"],
    )


class ResultIdentityTests(unittest.TestCase):
    def test_cross_process_result_writes_are_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "parallel_processes.csv")
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=4,
                mp_context=context,
            ) as executor:
                list(
                    executor.map(
                        _write_result_in_process,
                        [(path, index) for index in range(16)],
                    )
                )
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            {row["config_hash"] for row in rows},
            {str(index) for index in range(16)},
        )

    def test_concurrent_result_writes_are_serialized(self):
        header = ["config_hash", "metric"]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "parallel.csv")

            def write(index):
                append_csv(
                    path,
                    {"config_hash": str(index), "metric": float(index)},
                    header,
                    key_fields=["config_hash"],
                )

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(write, range(32)))
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 32)
        self.assertEqual(
            {row["config_hash"] for row in rows},
            {str(index) for index in range(32)},
        )

    def test_train_loss_selection_freezes_cpu_best_state(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        losses = iter([1.0, 2.0])

        def fake_epoch(active_model, *args, **kwargs):
            with torch.no_grad():
                active_model.weight.add_(1.0)
            return next(losses)

        initial = model.weight.detach().clone()
        with patch("run_triguard.train_one_epoch", side_effect=fake_epoch):
            train_model(
                model,
                train_loader=[],
                opt=optimizer,
                device=torch.device("cpu"),
                epochs=2,
                lambda_entropy=0.0,
                patience=1,
                selection_policy="train_loss",
            )
        self.assertTrue(torch.equal(model.weight, initial + 1.0))

    def test_checkpoint_resume_requires_weights_and_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            arguments = SimpleNamespace(
                save_ckpt=True,
                eval_only=False,
                checkpoint_path="",
                out=directory,
                lambda_wads=0.0,
                lambda_rar=0.0,
                lambda_far=0.0,
                lambda_curvature=0.0,
                lambda_robust=0.0,
                lambda_attr_mass=0.0,
                reference_risk="max",
                reference_distance="signed",
                reference_bank="",
            )
            setting = {
                "mode": "main",
                "dataset": "cifar10",
                "model_name": "resnet50",
                "seed": 0,
                "lam": 0.0,
                "config_hash": "abc123",
            }
            checkpoint = resolve_checkpoint_path(arguments, **setting)
            self.assertFalse(
                checkpoint_artifacts_complete(arguments, **setting)
            )
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            with open(checkpoint, "wb") as handle:
                handle.write(b"weights")
            self.assertFalse(
                checkpoint_artifacts_complete(arguments, **setting)
            )
            write_checkpoint_metadata(
                checkpoint,
                {
                    "config_hash": "abc123",
                    "code_hash": experiment_identity(
                        SimpleNamespace(
                            out=directory,
                            save_ckpt=False,
                        ),
                        mode="test",
                    )[1]["code_hash"],
                },
            )
            self.assertTrue(
                checkpoint_artifacts_complete(arguments, **setting)
            )
            self.assertIn("_cfgabc123.pt", checkpoint)

    def test_duplicate_seeds_and_unsupported_pairs_fail_early(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_seeds(0, "0,1,1")
        with self.assertRaisesRegex(ValueError, "Unsupported explicit"):
            get_experiment_grid("mnist", "convnext_tiny_imagenet", "workshop")

    def test_stale_protocol_outputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected only 2.3"):
            validate_protocol_values(["2.2"], "stale.csv")

    def test_calibration_sidecar_binds_checkpoint_to_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "calibration.pt")
            reservation = os.path.join(directory, "reservation.pt")
            torch_payload = b"checkpoint-state"
            with open(checkpoint, "wb") as handle:
                handle.write(torch_payload)
            with open(reservation, "wb") as handle:
                handle.write(b"reservation-indices")
            reservation_sha = hashlib.sha256(b"reservation-indices").hexdigest()
            write_checkpoint_metadata(
                checkpoint,
                {
                    "dataset": "cifar10",
                    "model": "resnet50_imagenet",
                    "training_reservation_sha256": reservation_sha,
                },
            )
            metadata = verify_calibration_checkpoint(
                checkpoint,
                reservation,
                "cifar10",
                "resnet50_imagenet",
            )
            self.assertEqual(
                metadata["training_reservation_sha256"], reservation_sha
            )
            with open(reservation, "wb") as handle:
                handle.write(b"different-reservation")
            with self.assertRaisesRegex(ValueError, "supplied candidate reservation"):
                verify_calibration_checkpoint(
                    checkpoint,
                    reservation,
                    "cifar10",
                    "resnet50_imagenet",
                )

    def test_run_condition_and_comparison_hashes_have_distinct_roles(self):
        base = {
            "out": "outputs/test",
            "seed": 0,
            "seeds": None,
            "dataset": "cifar10",
            "model": "resnet50_imagenet",
            "mode": "main",
            "save_ckpt": False,
            "load_ckpt": "",
            "reference_bank": "",
            "heldout_reference_bank": "",
            "lambda_entropy": 0.0,
            "lambda_wads": 0.0,
            "ig_steps": 32,
        }

        def identity(arguments, seed):
            return experiment_identity(
                SimpleNamespace(**arguments),
                mode="main",
                dataset="cifar10",
                model="resnet50_imagenet",
                seed=seed,
                lambda_entropy=arguments["lambda_entropy"],
            )

        run_zero, payload_zero = identity(base, 0)
        run_one, payload_one = identity(base, 1)
        self.assertRegex(payload_zero["code_hash"], r"^[0-9a-f]{16}$")
        self.assertNotEqual(run_zero, run_one)
        self.assertEqual(payload_zero["condition_hash"], payload_one["condition_hash"])

        treatment = {**base, "lambda_wads": 0.05}
        _, payload_treatment = identity(treatment, 0)
        self.assertNotEqual(
            payload_zero["condition_hash"], payload_treatment["condition_hash"]
        )
        self.assertEqual(
            payload_zero["comparison_hash"], payload_treatment["comparison_hash"]
        )

        protocol_change = {**base, "ig_steps": 64}
        _, payload_protocol_change = identity(protocol_change, 0)
        self.assertNotEqual(
            payload_zero["comparison_hash"],
            payload_protocol_change["comparison_hash"],
        )

    def test_duplicate_identity_is_rejected_without_appending(self):
        header = ["protocol_version", "config_hash", "metric"]
        row = {"protocol_version": "2.0", "config_hash": "abc", "metric": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.csv")
            append_csv(path, row, header, key_fields=["config_hash"])
            with self.assertRaisesRegex(ValueError, "Duplicate experiment row"):
                append_csv(path, row, header, key_fields=["config_hash"])
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(csv_contains_identity(path, {"config_hash": "abc"}))
            self.assertFalse(csv_contains_identity(path, {"config_hash": "missing"}))
        self.assertEqual(len(rows), 1)

    def test_duplicate_columns_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "results.csv")
            with self.assertRaisesRegex(ValueError, "duplicate columns"):
                append_csv(path, {"id": "x"}, ["id", "id"], key_fields=["id"])


if __name__ == "__main__":
    unittest.main()
