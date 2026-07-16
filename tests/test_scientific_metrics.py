import os
import copy
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


_REAL_VERSION_INFO = sys.version_info
sys.version_info = (3, 10, 12, "final", 0)
try:
    from triguard.attacks import clamp_input, uniform_like
    from triguard.attributions import (
        attribution_entropy,
        attribution_allocation_distance,
        completeness_orthogonal_distance,
        integrated_gradients,
        sanitize_attribution,
        smoothgrad_squared,
    )
    from triguard.faithfulness import deletion_insertion_curve
    from triguard.audit_pair_sampling import audit_pair_loss_matrix
    from triguard.eval import (
        _PixelSpaceModel,
        _baseline_drift_metrics,
        _stability_perturbations,
        autoattack_accuracy,
    )
    from triguard.references import (
        load_index_reservation,
        reference_source_indices,
        sample_reference_baselines,
    )
    from triguard.train import (
        _aggregate_reference_risk,
        _one_step_linf_perturb,
        baseline_regularization_terms,
        differentiable_integrated_gradients,
        differentiable_integrated_gradients_many,
        train_one_epoch,
        worst_baseline_drift_term,
    )
finally:
    sys.version_info = _REAL_VERSION_INFO


class QuadraticClassifier(torch.nn.Module):
    def forward(self, x):
        score = x.flatten(1).pow(2).sum(dim=1)
        return torch.stack([score, -score], dim=1)


class ChannelLinearClassifier(torch.nn.Module):
    def forward(self, x):
        score = x.flatten(2).sum(dim=2).sum(dim=1)
        return torch.stack([score, -score], dim=1)


class BatchNormClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(1)
        self.head = torch.nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.norm(x).flatten(1))


class TinyNonlinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 2),
        )

    def forward(self, x):
        return self.layers(x)


class ScientificMetricTests(unittest.TestCase):
    def test_smoothgrad_noise_samples_are_batched_without_metric_change(self):
        model = QuadraticClassifier()
        x = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]])
        sequential = smoothgrad_squared(
            model,
            x,
            target=0,
            noise_level=0.1,
            n_samples=7,
            generator=torch.Generator().manual_seed(23),
            sample_batch_size=1,
        )
        batched = smoothgrad_squared(
            model,
            x,
            target=0,
            noise_level=0.1,
            n_samples=7,
            generator=torch.Generator().manual_seed(23),
            sample_batch_size=4,
        )
        self.assertTrue(torch.allclose(sequential, batched, atol=1e-7))

    def test_deletion_curve_batches_model_evaluations(self):
        class CountingClassifier(ChannelLinearClassifier):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return super().forward(x)

        model = CountingClassifier()
        x = torch.arange(1.0, 7.0).reshape(1, 1, 2, 3)
        deletion_insertion_curve(
            model,
            x,
            0,
            x,
            "deletion",
            steps=5,
            baseline=torch.zeros_like(x),
            batch_size=64,
        )
        self.assertEqual(model.calls, 1)

    def test_regularizer_microbatch_matches_full_batch_update(self):
        torch.manual_seed(13)
        full_model = TinyNonlinearClassifier()
        micro_model = copy.deepcopy(full_model)
        x = torch.randn(4, 1, 2, 2)
        y = torch.tensor([0, 1, 0, 1])
        common = {
            "lambda_entropy": 0.0,
            "lambda_wads": 0.1,
            "lambda_attr_mass": 0.07,
            "attr_mass_floor": 0.9,
            "triguard_ig_steps": 2,
            "baseline_modes": "zero,midpoint",
            "baseline_min": 0.0,
            "baseline_max": 1.0,
        }
        full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.01)
        micro_optimizer = torch.optim.SGD(micro_model.parameters(), lr=0.01)
        train_one_epoch(
            full_model,
            [(x, y)],
            full_optimizer,
            torch.device("cpu"),
            regularizer_microbatch=0,
            **common,
        )
        train_one_epoch(
            micro_model,
            [(x, y)],
            micro_optimizer,
            torch.device("cpu"),
            regularizer_microbatch=1,
            **common,
        )
        for full_parameter, micro_parameter in zip(
            full_model.parameters(), micro_model.parameters()
        ):
            self.assertTrue(
                torch.allclose(
                    full_parameter, micro_parameter, atol=1e-7, rtol=1e-6
                )
            )

    def test_vectorized_reference_ig_matches_sequential_and_checkpointed(self):
        torch.manual_seed(3)
        model = TinyNonlinearClassifier()
        x = torch.randn(2, 1, 2, 2)
        targets = torch.tensor([0, 1])
        baselines = {
            "zero": torch.zeros_like(x),
            "one": torch.ones_like(x),
            "mid": torch.full_like(x, 0.5),
        }
        sequential = {
            name: differentiable_integrated_gradients(
                model, x, targets, baseline, steps=3
            )
            for name, baseline in baselines.items()
        }
        vectorized = differentiable_integrated_gradients_many(
            model, x, targets, baselines, steps=3
        )
        checkpointed = differentiable_integrated_gradients_many(
            model,
            x,
            targets,
            baselines,
            steps=3,
            checkpoint_activations=True,
        )
        for name in baselines:
            self.assertTrue(
                torch.allclose(sequential[name], vectorized[name], atol=1e-6)
            )
            self.assertTrue(
                torch.allclose(vectorized[name], checkpointed[name], atol=1e-6)
            )

        loss = sum(value.square().mean() for value in checkpointed.values())
        loss.backward()
        self.assertIsNotNone(model.layers[1].weight.grad)

    def test_pair_sampling_audit_detects_max_risk_downward_bias(self):
        pair_losses = np.asarray(
            [
                [0.1, 0.2, 0.3, 1.0],
                [0.2, 0.4, 0.6, 1.2],
                [0.1, 0.5, 0.7, 1.4],
            ]
        )
        result = audit_pair_loss_matrix(
            pair_losses,
            sample_size=2,
            risk="max",
            cvar_alpha=0.75,
            trials=4000,
            seed=17,
        )
        self.assertLess(float(result["bias"]), -0.05)
        exact = audit_pair_loss_matrix(
            pair_losses,
            sample_size=4,
            risk="max",
            cvar_alpha=0.75,
            trials=20,
            seed=17,
        )
        self.assertAlmostEqual(float(exact["bias"]), 0.0)
        self.assertAlmostEqual(float(exact["rank_correlation"]), 1.0)

    def test_mass_floor_reference_scores_use_one_batched_forward(self):
        class CountingClassifier(ChannelLinearClassifier):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, x):
                self.calls += 1
                return super().forward(x)

        model = CountingClassifier()
        x = torch.ones(2, 1, 1, 2)
        family = {
            "a": torch.zeros_like(x),
            "b": torch.full_like(x, 0.25),
            "c": torch.full_like(x, 0.5),
        }
        with mock.patch(
            "triguard.train.make_baseline_family",
            return_value=family,
        ), mock.patch(
            "triguard.train.differentiable_integrated_gradients",
            side_effect=lambda model, x, y, baseline, **kwargs: x - baseline,
        ):
            baseline_regularization_terms(
                model,
                x,
                torch.tensor([0, 1]),
                baseline_modes="a,b,c",
                baseline_min=0.0,
                baseline_max=1.0,
                vectorized_reference_ig=False,
            )
        self.assertEqual(model.calls, 1)

    def test_pair_sampling_happens_before_reference_ig(self):
        x = torch.zeros(2, 1, 1, 2)
        family = {
            "a": torch.full_like(x, 0.0),
            "b": torch.full_like(x, 0.2),
            "c": torch.full_like(x, 0.4),
            "d": torch.full_like(x, 0.6),
            "e": torch.full_like(x, 0.8),
        }
        captured = {}

        def fake_many(model, active_x, targets, baselines, **kwargs):
            captured["names"] = list(baselines)
            return baselines

        torch.manual_seed(7)
        with mock.patch(
            "triguard.train.make_baseline_family", return_value=family
        ), mock.patch(
            "triguard.train.differentiable_integrated_gradients_many",
            side_effect=fake_many,
        ):
            worst_baseline_drift_term(
                model=object(),
                x=x,
                y=torch.tensor([0, 1]),
                baseline_modes="a,b,c,d,e",
                baseline_min=0.0,
                baseline_max=1.0,
                reference_pair_samples=1,
                vectorized_reference_ig=True,
            )
        self.assertEqual(len(captured["names"]), 2)

    def test_sampled_mass_penalty_reuses_pair_references(self):
        x = torch.ones(2, 1, 1, 2)
        family = {
            "a": torch.full_like(x, 0.0),
            "b": torch.full_like(x, 0.2),
            "c": torch.full_like(x, 0.4),
            "d": torch.full_like(x, 0.6),
            "e": torch.full_like(x, 0.8),
        }
        captured = {}

        def fake_many(model, active_x, targets, baselines, **kwargs):
            captured["names"] = list(baselines)
            return baselines

        torch.manual_seed(9)
        with mock.patch(
            "triguard.train.make_baseline_family", return_value=family
        ), mock.patch(
            "triguard.train.differentiable_integrated_gradients_many",
            side_effect=fake_many,
        ):
            baseline_regularization_terms(
                ChannelLinearClassifier(),
                x,
                torch.tensor([0, 1]),
                baseline_modes="a,b,c,d,e",
                baseline_min=0.0,
                baseline_max=1.0,
                reference_pair_samples=1,
                sampled_mass_penalty=True,
                vectorized_reference_ig=True,
            )
        self.assertEqual(len(captured["names"]), 2)

    def test_regularizer_microbatch_keeps_one_optimizer_step(self):
        torch.manual_seed(11)
        model = TinyNonlinearClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = [
            (
                torch.randn(4, 1, 2, 2),
                torch.tensor([0, 1, 0, 1]),
            )
        ]
        with mock.patch.object(
            optimizer, "step", wraps=optimizer.step
        ) as step:
            train_one_epoch(
                model,
                loader,
                optimizer,
                torch.device("cpu"),
                lambda_entropy=0.05,
                regularizer_microbatch=1,
            )
        self.assertEqual(step.call_count, 1)

    def test_attribution_regularizer_does_not_update_batchnorm_statistics(self):
        model = BatchNormClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loader = [
            (
                torch.tensor(
                    [
                        [[[0.0, 0.2], [0.4, 0.6]]],
                        [[[0.1, 0.3], [0.5, 0.7]]],
                    ]
                ),
                torch.tensor([0, 1]),
            )
        ]
        train_one_epoch(
            model,
            loader,
            optimizer,
            torch.device("cpu"),
            lambda_entropy=0.0,
            lambda_wads=0.1,
            triguard_ig_steps=2,
            baseline_modes="zero,midpoint",
            baseline_min=0.0,
            baseline_max=1.0,
        )
        self.assertEqual(int(model.norm.num_batches_tracked.item()), 1)
        self.assertTrue(model.training)

    def test_spatial_shift_uses_padding_instead_of_wraparound(self):
        x = torch.arange(9.0).view(1, 1, 3, 3)
        shifted = _stability_perturbations(
            x,
            modes="shift",
            clamp_min=0.0,
            clamp_max=8.0,
            noise_std=0.0,
        )[0][1]
        expected = torch.tensor([[[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [3.0, 3.0, 4.0]]]])
        self.assertTrue(torch.equal(shifted, expected))

    def test_autoattack_receives_seed_at_construction(self):
        captured = {}

        class FakeAutoAttack:
            def __init__(self, model, **kwargs):
                self.model = model
                captured.update(kwargs)

            def run_standard_evaluation(self, images, labels, bs):
                captured["batch"] = bs
                return images

        module = types.ModuleType("autoattack")
        module.AutoAttack = FakeAutoAttack
        loader = [(torch.ones(2, 1, 2, 2), torch.zeros(2, dtype=torch.long))]
        with mock.patch.dict(sys.modules, {"autoattack": module}):
            accuracy, count = autoattack_accuracy(
                ChannelLinearClassifier(),
                loader,
                torch.device("cpu"),
                pixel_eps=0.1,
                normalization_mean=(0.0,),
                normalization_std=(1.0,),
                max_samples=2,
                batch_size=2,
                seed=19,
            )
        self.assertEqual(captured["seed"], 19)
        self.assertEqual(captured["version"], "standard")
        self.assertEqual(count, 2)
        self.assertEqual(accuracy, 1.0)

    def test_index_reservation_round_trip_is_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "reservation.pt")
            torch.save(
                {
                    "indices": torch.tensor([1, 3, 5, 7]),
                    "metadata": {
                        "dataset": "cifar10",
                        "input_profile": "imagenet",
                        "source_split": "train",
                        "selection": "model_independent_random_reservation",
                        "seed": 0,
                    },
                },
                path,
            )
            indices, metadata = load_index_reservation(path)
        self.assertEqual(indices, [1, 3, 5, 7])
        self.assertEqual(metadata["dataset"], "cifar10")

    def test_reservation_without_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "reservation.pt")
            torch.save(
                {
                    "indices": torch.tensor([1, 3, 5, 7]),
                    "metadata": {"dataset": "cifar10"},
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "metadata is missing"):
                load_index_reservation(path)

    def test_reference_bank_metadata_matches_tensor_and_role(self):
        metadata = {
            "dataset": "cifar10",
            "input_profile": "imagenet",
            "source_split": "train",
            "role": "heldout",
            "source_indices": [2, 4],
        }
        self.assertEqual(
            reference_source_indices(metadata, 2, "heldout"), [2, 4]
        )
        with self.assertRaisesRegex(ValueError, "Expected a train"):
            reference_source_indices(metadata, 2, "train")

    def test_reference_bank_sampling_is_without_replacement_per_example(self):
        bank = torch.arange(5.0).view(5, 1, 1, 1)
        x = torch.zeros(2, 1, 1, 1)
        generator = torch.Generator().manual_seed(17)
        sampled = sample_reference_baselines(x, bank, count=5, generator=generator)

        values = torch.stack(list(sampled.values()), dim=1).flatten(1)
        expected = torch.arange(5.0).expand(2, -1)
        self.assertTrue(torch.equal(values.sort(dim=1).values, expected))

    def test_mass_ratio_matches_the_anti_collapse_quantity(self):
        metrics = _baseline_drift_metrics(
            ChannelLinearClassifier(),
            torch.ones(1, 1, 2, 2),
            target=0,
            modes="zero,midpoint",
            baseline_min=0.0,
            baseline_max=1.0,
            steps=4,
        )
        self.assertAlmostEqual(metrics["baseline_attr_mass_ratio_min"], 1.0)

    def test_integrated_gradients_uses_stated_right_endpoint_sum(self):
        model = QuadraticClassifier()
        x = torch.ones(1, 1, 1, 1)
        baseline = torch.zeros_like(x)
        attr = integrated_gradients(model, x, 0, baseline, steps=2)
        # Gradients at alpha={1/2, 1} average to 1.5. The previous left
        # endpoint implementation returned 0.5 for this example.
        self.assertTrue(torch.allclose(attr, torch.tensor([[[[1.5]]]])))

    def test_allocation_distance_is_scale_free_and_bounded(self):
        left = torch.tensor([[[[2.0, -1.0]]]])
        same_allocation = 10.0 * left
        opposite = -left
        self.assertTrue(
            torch.allclose(
                attribution_allocation_distance(left, same_allocation),
                torch.zeros(1),
            )
        )
        self.assertTrue(
            torch.allclose(
                attribution_allocation_distance(left, opposite),
                torch.tensor([2.0]),
            )
        )

    def test_wads_takes_per_sample_max_before_batch_mean(self):
        # Sample 1 has pair (a,b) as its worst pair; sample 2 has (a,c).
        a = torch.tensor([[[[1.0, 0.0]]], [[[1.0, 0.0]]]])
        b = torch.tensor([[[[0.0, 1.0]]], [[[0.5, 0.5]]]])
        c = torch.tensor([[[[0.5, 0.5]]], [[[0.0, 1.0]]]])
        family = {"a": a, "b": b, "c": c}

        with mock.patch("triguard.train.make_baseline_family", return_value=family), mock.patch(
            "triguard.train.differentiable_integrated_gradients",
            side_effect=lambda model, x, y, baseline, **kwargs: baseline,
        ):
            loss = worst_baseline_drift_term(
                model=object(),
                x=torch.zeros_like(a),
                y=torch.tensor([0, 0]),
                baseline_modes="a,b,c",
                baseline_min=0.0,
                baseline_max=1.0,
                vectorized_reference_ig=False,
            )
        self.assertAlmostEqual(float(loss), 2.0)

    def test_channelwise_clamp_and_uniform_sampling_respect_each_channel(self):
        x = torch.tensor([[[[-10.0]], [[0.5]], [[10.0]]]])
        lower = (-2.0, -1.0, 0.0)
        upper = (-1.0, 1.0, 2.0)
        clamped = clamp_input(x, lower, upper)
        expected = torch.tensor([[[[-2.0]], [[0.5]], [[2.0]]]])
        self.assertTrue(torch.allclose(clamped, expected))

        samples = uniform_like(torch.empty(128, 3, 1, 1), lower, upper)
        for channel, (lo, hi) in enumerate(zip(lower, upper)):
            self.assertGreaterEqual(float(samples[:, channel].min()), lo)
            self.assertLessEqual(float(samples[:, channel].max()), hi)

        scalar_clamped = clamp_input(torch.tensor([-2.0, 0.5, 3.0]), 0.0, 1.0)
        self.assertTrue(
            torch.allclose(scalar_clamped, torch.tensor([0.0, 0.5, 1.0]))
        )

    def test_training_perturbation_preserves_channelwise_pixel_budget(self):
        x = torch.zeros(1, 3, 2, 2)
        eps = (0.05, 0.10, 0.20)
        perturbed = _one_step_linf_perturb(
            ChannelLinearClassifier(),
            x,
            torch.tensor([1]),
            eps=eps,
            alpha=(1.0, 1.0, 1.0),
            clamp_min=(-1.0, -1.0, -1.0),
            clamp_max=(1.0, 1.0, 1.0),
        )
        delta = (perturbed - x).abs()
        for channel, budget in enumerate(eps):
            self.assertLessEqual(float(delta[:, channel].max()), budget + 1e-7)

    def test_completeness_orthogonal_distance_removes_only_constant_component(self):
        left = torch.tensor([[[[1.0, 2.0, 4.0]]]])
        right = torch.tensor([[[[0.0, 2.0, 1.0]]]])
        difference = (left - right).flatten(1)
        residual = difference - difference.mean(dim=1, keepdim=True)
        expected = residual.square().mean(dim=1).sqrt()
        self.assertTrue(
            torch.allclose(completeness_orthogonal_distance(left, right), expected)
        )
        self.assertTrue(
            torch.allclose(
                completeness_orthogonal_distance(left, left + 5.0),
                torch.zeros(1),
            )
        )

    def test_pixel_entropy_is_channel_and_dimension_normalized(self):
        gray = torch.tensor([[[[1.0, 3.0], [0.0, 0.0]]]])
        rgb = gray.repeat(1, 3, 1, 1)
        gray_entropy = attribution_entropy(
            gray, normalized=True, pixel_level=True
        )
        rgb_entropy = attribution_entropy(rgb, normalized=True, pixel_level=True)
        self.assertAlmostEqual(gray_entropy, rgb_entropy)
        self.assertGreaterEqual(gray_entropy, 0.0)
        self.assertLessEqual(gray_entropy, 1.0)

    def test_nonfinite_attribution_fails_loudly(self):
        with self.assertRaises(FloatingPointError):
            sanitize_attribution(torch.tensor([float("nan")]))

    def test_faithfulness_curve_reaches_exact_endpoints_with_remainder_pixels(self):
        model = ChannelLinearClassifier()
        x = torch.arange(1.0, 7.0).reshape(1, 1, 2, 3)
        baseline = torch.zeros_like(x)
        attr = x.clone()
        deletion = deletion_insertion_curve(
            model, x, 0, attr, "deletion", steps=4, baseline=baseline
        )
        insertion = deletion_insertion_curve(
            model, x, 0, attr, "insertion", steps=4, baseline=baseline
        )
        with torch.no_grad():
            input_probability = torch.softmax(model(x), dim=1)[0, 0].item()
            baseline_probability = torch.softmax(model(baseline), dim=1)[0, 0].item()
        self.assertAlmostEqual(deletion[-1], baseline_probability)
        self.assertAlmostEqual(insertion[-1], input_probability)

    def test_cvar_reference_risk_uses_upper_tail_per_sample(self):
        losses = torch.tensor([[1.0, 10.0], [2.0, 20.0], [9.0, 30.0], [8.0, 40.0]])
        value = _aggregate_reference_risk(losses, "cvar", cvar_alpha=0.5)
        # Per-sample top halves are means (8.5, 35), then averaged over samples.
        self.assertAlmostEqual(float(value), 21.75)

    def test_pixel_space_adapter_recovers_normalized_model_input(self):
        model = ChannelLinearClassifier()
        adapter = _PixelSpaceModel(
            model, mean=(0.5, 0.25, 0.75), std=(0.5, 0.25, 0.25)
        )
        pixels = torch.tensor([[[[0.5]], [[0.5]], [[1.0]]]])
        normalized = torch.tensor([[[[0.0]], [[1.0]], [[1.0]]]])
        self.assertTrue(torch.allclose(adapter(pixels), model(normalized)))


if __name__ == "__main__":
    unittest.main()
