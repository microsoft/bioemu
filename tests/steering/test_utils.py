"""Tests for bioemu.steering.utils — resampling, loss functions, and helpers."""

import torch
from torch_geometric.data import Batch, Data


class TestStratifiedResample:
    """Tests for stratified_resample."""

    def test_uniform_weights_spread(self):
        """Uniform weights should produce indices that cover most of the range."""
        from bioemu.steering.utils import stratified_resample

        torch.manual_seed(0)
        B, N = 1, 100
        weights = torch.ones(B, N) / N
        idx = stratified_resample(weights)
        assert idx.shape == (B, N)
        unique = idx[0].unique()
        assert unique.numel() >= N - 5

    def test_delta_distribution(self):
        """All weight on particle k → all indices should be k."""
        from bioemu.steering.utils import stratified_resample

        B, N, k = 2, 20, 7
        weights = torch.zeros(B, N)
        weights[:, k] = 1.0
        idx = stratified_resample(weights)
        assert (idx == k).all()


class TestResampleBasedOnLogWeights:
    """Tests for resample_based_on_log_weights."""

    @staticmethod
    def _make_batch(n_samples: int, n_residues: int = 5) -> Batch:
        data_list = []
        for i in range(n_samples):
            d = Data(
                pos=torch.randn(n_residues, 3),
                node_orientations=torch.eye(3).unsqueeze(0).expand(n_residues, -1, -1).clone(),
                batch=torch.zeros(n_residues, dtype=torch.long),
                sequence=["ACDEF"],
            )
            data_list.append(d)
        return Batch.from_data_list(data_list)

    def test_equal_weights_no_resample(self):
        """When all weights are equal, ESS ≈ 1 → no resampling if threshold < 1."""
        from bioemu.steering.utils import resample_based_on_log_weights

        n = 16
        batch = self._make_batch(n)
        log_w = torch.zeros(n)
        new_batch, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=n,
            is_last_step=False,
            ess_threshold=0.5,
            step=0,
            t=0.5,
        )
        assert ess > 0.9
        torch.testing.assert_close(indices, torch.arange(n))

    def test_dominant_weight_resamples(self):
        """When one weight dominates and ESS is low, resampling should happen."""
        from bioemu.steering.utils import resample_based_on_log_weights

        n = 16
        batch = self._make_batch(n)
        log_w = torch.full((n,), -100.0)
        log_w[3] = 0.0
        new_batch, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=n,
            is_last_step=False,
            ess_threshold=0.5,
            step=0,
            t=0.5,
        )
        assert ess < 0.2
        torch.testing.assert_close(new_lw, torch.zeros(n))

    def test_last_step_always_resamples(self):
        """On the last step, resampling should always happen regardless of ESS."""
        from bioemu.steering.utils import resample_based_on_log_weights

        n = 8
        batch = self._make_batch(n)
        log_w = torch.zeros(n)
        new_batch, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=n,
            is_last_step=True,
            ess_threshold=0.5,
            step=99,
            t=0.001,
        )
        torch.testing.assert_close(new_lw, torch.zeros(n))

    def test_small_batch_treated_as_single_group(self):
        """When batch_size < n_particles, entire batch is treated as one group."""
        from bioemu.steering.utils import resample_based_on_log_weights

        n = 4
        batch = self._make_batch(n)
        log_w = torch.zeros(n)
        new_batch, new_lw, indices, ess = resample_based_on_log_weights(
            batch=batch,
            log_weight=log_w,
            n_particles=100,
            is_last_step=False,
            ess_threshold=0.5,
            step=0,
            t=0.5,
        )
        assert ess > 0.9


class TestComputeSequenceAlignment:
    """Tests for compute_sequence_alignment."""

    def test_identical_sequences(self):
        from bioemu.steering.utils import compute_sequence_alignment

        mapping = compute_sequence_alignment("ACDEF", "ACDEF")
        assert mapping == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    def test_subsequence(self):
        from bioemu.steering.utils import compute_sequence_alignment

        mapping = compute_sequence_alignment("ACE", "ACDEF")
        assert 0 in mapping  # A
        assert 1 in mapping  # C
        assert 2 in mapping  # E

    def test_different_sequences(self):
        """Completely different sequences should still return a dict."""
        from bioemu.steering.utils import compute_sequence_alignment

        mapping = compute_sequence_alignment("AAAA", "CCCC")
        assert isinstance(mapping, dict)


class TestGetPos0Rot0:
    """Tests for get_pos0_rot0 helper."""

    def test_identity_score_returns_input(self):
        """With zero score, x0 prediction should roughly match x_t / alpha_t."""
        from bioemu.steering.utils import _get_x0_given_xt_and_score

        # Simple test: x0 = (x + sigma^2 * score) / alpha
        x = torch.randn(10, 3)
        score = torch.zeros_like(x)
        batch_idx = torch.zeros(10, dtype=torch.long)

        # Mock SDE that returns alpha=1, sigma=0
        class MockSDE:
            def mean_coeff_and_std(self, x, t, batch_idx):
                return torch.ones_like(x), torch.zeros_like(x)

        x0 = _get_x0_given_xt_and_score(MockSDE(), x, torch.tensor([0.5]), batch_idx, score)
        torch.testing.assert_close(x0, x)
