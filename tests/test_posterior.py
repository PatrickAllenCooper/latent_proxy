from __future__ import annotations

import numpy as np
import pytest

from src.utils.posterior import (
    GaussianPosterior,
    ParticlePosterior,
    PosteriorBase,
    _clamp_theta,
)


class TestPosteriorBase:
    """Both implementations comply with the PosteriorBase interface."""

    @pytest.fixture(params=["gaussian", "particle"])
    def posterior(self, request: pytest.FixtureRequest) -> PosteriorBase:
        if request.param == "gaussian":
            return GaussianPosterior()
        return ParticlePosterior(n_particles=500)

    def test_has_mean(self, posterior: PosteriorBase) -> None:
        assert posterior.mean.shape == (3,)

    def test_has_variance(self, posterior: PosteriorBase) -> None:
        assert posterior.variance.shape == (3,)
        assert np.all(posterior.variance > 0)

    def test_has_std(self, posterior: PosteriorBase) -> None:
        assert posterior.std.shape == (3,)
        np.testing.assert_allclose(posterior.std, np.sqrt(posterior.variance))

    def test_sample_shape(self, posterior: PosteriorBase) -> None:
        samples = posterior.sample(100, rng=np.random.default_rng(0))
        assert samples.shape == (100, 3)

    def test_samples_in_valid_range(self, posterior: PosteriorBase) -> None:
        samples = posterior.sample(500, rng=np.random.default_rng(0))
        assert np.all(samples[:, 0] > 0)
        assert np.all(samples[:, 0] <= 1.0)
        assert np.all(samples[:, 1] >= 0)
        assert np.all(samples[:, 2] >= 1.0)

    def test_entropy_is_finite(self, posterior: PosteriorBase) -> None:
        h = posterior.entropy()
        assert np.isfinite(h)

    def test_to_dict(self, posterior: PosteriorBase) -> None:
        d = posterior.to_dict()
        assert "gamma" in d
        assert "alpha" in d
        assert "lambda_" in d

    def test_credible_region(self, posterior: PosteriorBase) -> None:
        region = posterior.credible_region(0.9, n_samples=1000,
                                          rng=np.random.default_rng(0))
        for name in ["gamma", "alpha", "lambda_"]:
            lo, hi = region[name]
            assert lo < hi

    def test_converged_with_tight_thresholds(self, posterior: PosteriorBase) -> None:
        result = posterior.converged({
            "gamma_variance_threshold": 1000.0,
            "alpha_variance_threshold": 1000.0,
            "lambda__variance_threshold": 1000.0,
        })
        assert result is True

    def test_not_converged_with_zero_thresholds(self, posterior: PosteriorBase) -> None:
        result = posterior.converged({
            "gamma_variance_threshold": 0.0,
        })
        assert result is False


class TestParticlePosterior:
    def test_initialization(self) -> None:
        pp = ParticlePosterior(n_particles=200)
        assert pp.particles.shape == (200, 3)
        np.testing.assert_allclose(pp.weights.sum(), 1.0)

    def test_effective_sample_size(self) -> None:
        pp = ParticlePosterior(n_particles=100)
        ess = pp.effective_sample_size
        assert 90 < ess <= 100

    def test_ess_drops_with_skewed_weights(self) -> None:
        pp = ParticlePosterior(n_particles=100)
        pp.weights = np.zeros(100)
        pp.weights[0] = 1.0
        assert pp.effective_sample_size == pytest.approx(1.0)

    def test_systematic_resample_resets_weights(self) -> None:
        pp = ParticlePosterior(n_particles=100)
        pp.weights = np.zeros(100)
        pp.weights[0] = 1.0
        pp._systematic_resample()
        np.testing.assert_allclose(pp.weights, np.ones(100) / 100)


class TestParticlePosteriorUpdate:
    def test_update_contracts_posterior(self) -> None:
        pp = ParticlePosterior(n_particles=500)
        initial_var = pp.variance.copy()

        option_a = np.array([0.6, 0.3, 0.05, 0.05])
        option_b = np.array([0.05, 0.15, 0.4, 0.4])
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.0001, 0.0016, 0.01, 0.04])

        for _ in range(5):
            pp.update_from_choice(
                choice=0,
                option_a_alloc=option_a,
                option_b_alloc=option_b,
                channel_means=means,
                channel_variances=variances,
                current_wealth=1000.0,
                rounds_remaining=10,
                temperature=0.1,
            )

        assert np.any(pp.variance < initial_var), (
            "Posterior should contract after observations"
        )


class TestGaussianPosteriorUpdate:
    def test_kalman_update_contracts(self) -> None:
        gp = GaussianPosterior()
        initial_var = gp.variance.copy()
        gp.update(np.array([0.8, 1.5, 2.0]), observation_variance=0.05)
        assert np.all(gp.variance < initial_var)

    def test_update_from_choice_changes_mean(self) -> None:
        gp = GaussianPosterior()
        initial_mean = gp.mean.copy()

        option_a = np.array([0.6, 0.3, 0.05, 0.05])
        option_b = np.array([0.05, 0.15, 0.4, 0.4])
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.0001, 0.0016, 0.01, 0.04])

        gp.update_from_choice(
            choice=0,
            option_a_alloc=option_a,
            option_b_alloc=option_b,
            channel_means=means,
            channel_variances=variances,
            current_wealth=1000.0,
            rounds_remaining=10,
            temperature=0.1,
        )

        assert not np.allclose(gp.mean, initial_mean), "Mean should shift"


class TestGaussianParticleAgreement:
    def test_both_shift_toward_true_theta(self) -> None:
        true_theta = np.array([0.9, 0.5, 1.2])
        option_a = np.array([0.6, 0.3, 0.05, 0.05])
        option_b = np.array([0.05, 0.15, 0.4, 0.4])
        means = np.array([0.02, 0.06, 0.12, 0.08])
        variances = np.array([0.0001, 0.0016, 0.01, 0.04])

        gp = GaussianPosterior()
        pp = ParticlePosterior(n_particles=500)

        for _ in range(3):
            gp.update_from_choice(
                0, option_a, option_b, means, variances, 1000.0, 10, 0.1,
            )
            pp.update_from_choice(
                0, option_a, option_b, means, variances, 1000.0, 10, 0.1,
            )

        gp_dist = np.linalg.norm(gp.mean - true_theta)
        pp_dist = np.linalg.norm(pp.mean - true_theta)

        assert gp_dist < 2.0, f"Gaussian mean too far: {gp_dist}"
        assert pp_dist < 2.0, f"Particle mean too far: {pp_dist}"


class TestClampTheta:
    def test_clamps_gamma(self) -> None:
        t = np.array([-0.5, 1.0, 1.5])
        clamped = _clamp_theta(t)
        assert clamped[0] > 0

    def test_clamps_lambda(self) -> None:
        t = np.array([0.5, 1.0, 0.5])
        clamped = _clamp_theta(t)
        assert clamped[2] >= 1.0

    def test_batch(self) -> None:
        batch = np.array([[-0.5, -1.0, 0.0], [0.5, 1.0, 2.0]])
        clamped = _clamp_theta(batch)
        assert clamped.shape == (2, 3)
        assert np.all(clamped[:, 0] > 0)
        assert np.all(clamped[:, 2] >= 1.0)
