# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import numpy as np
import pytest
import torch
from scipy.integrate import cumulative_trapezoid
from scipy.spatial.transform import Rotation

from bioemu.so3_sde import (
    SampleIGSO3,
    SampleUSO3,
    ScoreSO3,
    _broadcast_identity,
    _rotquat_to_axis_angle,
    angle_from_rotmat,
    apply_rotvec_to_rotmat,
    digso3_expansion,
    dlog_igso3_expansion,
    igso3_expansion,
    integrate_trapezoid_cumulative,
    rotmat_to_rotvec,
    rotquat_to_rotmat,
    rotquat_to_rotvec,
    rotvec_to_rotmat,
    scale_rotmat,
    skew_matrix_exponential_map,
    skew_matrix_exponential_map_axis_angle,
    skew_matrix_to_vector,
    vector_to_skew_matrix,
)

NUM_ROTATIONS = 10
NUM_OMEGA = 500
NUM_SIGMA = 500
NUM_SAMPLES = 10


@pytest.fixture
def rotation_angles(num_rotations=NUM_ROTATIONS):
    """Generate rotation angles in the interval [5e-2, pi-1e-2]."""
    angles = np.random.uniform(5e-2, np.pi - 1e-2, size=(num_rotations,))
    angles = torch.from_numpy(angles)
    return angles


@pytest.fixture
def rotation_vectors(rotation_angles):
    """
    Create rotation vectors by uniformly sampling rotation axes and scaling their length to the
    target rotation angles.
    """
    vectors = np.random.normal(size=(rotation_angles.shape[0], 3))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = torch.from_numpy(vectors)
    rotvecs = rotation_angles[:, None] * vectors
    return rotvecs


@pytest.fixture
def rotation_matrices(rotation_vectors):
    """Convert rotation vectors to matrices using Scipy."""
    rotation = Rotation.from_rotvec(rotation_vectors)
    rotmats = rotation.as_matrix()
    rotmats = torch.from_numpy(rotmats)
    return rotmats


@pytest.fixture
def rotation_quaternions(rotation_vectors):
    """Convert rotation vectors to quaternions using Scipy."""
    rotation = Rotation.from_rotvec(rotation_vectors)
    rotquats = rotation.as_quat()
    rotquats = torch.from_numpy(rotquats)
    # Scipy uses the convention [i,j,k,r] for quaternions, shift to [r,i,j,k].
    rotquats = torch.roll(rotquats, 1, dims=-1)
    return rotquats


@pytest.fixture
def skew_matrices(rotation_vectors):
    """Convert rotation vectors to skew matrix form."""
    skewmats = torch.zeros(*rotation_vectors.shape, 3, dtype=rotation_vectors.dtype)
    skewmats[..., 2, 1] = rotation_vectors[..., 0]
    skewmats[..., 0, 2] = rotation_vectors[..., 1]
    skewmats[..., 1, 0] = rotation_vectors[..., 2]
    skewmats = skewmats - skewmats.transpose(-2, -1)
    return skewmats


def test_integrate_trapezoid_cumulative() -> None:
    """Test for cumulative distribution function integration."""
    # Generate unequally spaced grid from 0 to 10.
    x = torch.linspace(0.0, 1.0, 100) ** 3
    x = x[None, :] * 10.0

    # Compute sinc function on grid.
    y = torch.sinc(x)

    int_y = integrate_trapezoid_cumulative(y, x)

    # Get Scipy reference and bring to torch conventions.
    int_y_ref = cumulative_trapezoid(y[0].numpy(), x[0].numpy())
    int_y_ref = torch.from_numpy(int_y_ref)[None, :]

    assert torch.allclose(int_y, int_y_ref)


def test_broadcast_identity(rotation_matrices: torch.Tensor) -> None:
    """Test identity matrix broadcasting."""
    id3 = _broadcast_identity(rotation_matrices)

    # Check shape.
    assert id3.shape == rotation_matrices.shape

    # Check diagonals are 1.
    diag = torch.diagonal(id3, dim1=-2, dim2=-1)
    assert torch.allclose(
        diag, torch.tensor(1.0, dtype=rotation_matrices.dtype, device=rotation_matrices.device)
    )

    # Check offdiagonals are 0
    offdiag = id3[..., torch.eye(3) != 1]
    assert torch.allclose(
        offdiag, torch.tensor(0.0, dtype=rotation_matrices.dtype, device=rotation_matrices.device)
    )


def test_vector_to_skew_matrix(rotation_vectors: torch.Tensor, skew_matrices: torch.Tensor) -> None:
    """Test construction of skew matrices from vectors."""
    skewmats = vector_to_skew_matrix(rotation_vectors)
    assert torch.allclose(skewmats, skew_matrices)


def test_skew_matrix_to_vector(rotation_vectors: torch.Tensor, skew_matrices: torch.Tensor) -> None:
    """Test extraction of vectors from skew matrices."""
    rotvecs = skew_matrix_to_vector(skew_matrices)
    assert torch.allclose(rotvecs, rotation_vectors)


def test_skew_matrix_exponential_map_axis_angle(
    rotation_vectors: torch.Tensor, rotation_matrices: torch.Tensor
) -> None:
    """Test skew matrix exponential computation."""
    angles = torch.norm(rotation_vectors, dim=-1)
    vectors = rotation_vectors / angles[..., None]

    # Exponentials using Rodrigues' formula.
    skewmats = vector_to_skew_matrix(vectors)
    exp_map = skew_matrix_exponential_map_axis_angle(angles, skewmats)

    # Reference matrix exponential
    exp_ref = torch.linalg.matrix_exp(angles[..., None, None] * skewmats)

    assert torch.allclose(exp_map, exp_ref)
    assert torch.allclose(exp_map, rotation_matrices)


def test_skew_matrix_exponential_map(
    rotation_vectors: torch.Tensor, rotation_matrices: torch.Tensor
) -> None:
    """Test skew matrix exponential computation."""
    angles = torch.norm(rotation_vectors, dim=-1)

    # Exponentials using Rodrigues' formula.
    skewmats = vector_to_skew_matrix(rotation_vectors)
    exp_map = skew_matrix_exponential_map(angles, skewmats)

    # Reference matrix exponential
    exp_ref = torch.linalg.matrix_exp(skewmats)

    assert torch.allclose(exp_map, exp_ref)
    assert torch.allclose(exp_map, rotation_matrices)


def test_angle_from_rotmat(rotation_angles: torch.Tensor, rotation_matrices: torch.Tensor) -> None:
    """Test computation of rotation angles from rotation matrices."""
    angles, angles_sin, angles_cos = angle_from_rotmat(rotation_matrices)
    assert torch.allclose(angles, rotation_angles)
    assert torch.allclose(angles_sin, torch.sin(rotation_angles))
    assert torch.allclose(angles_cos, torch.cos(rotation_angles))


def test_rotvec_to_rotmat(rotation_vectors: torch.Tensor, rotation_matrices: torch.Tensor) -> None:
    """Test conversion of rotation vectors to rotation matrices."""
    rotmats = rotvec_to_rotmat(rotation_vectors)
    assert torch.allclose(rotmats, rotation_matrices)


def test_rotmat_to_rotvec(rotation_vectors: torch.Tensor, rotation_matrices: torch.Tensor) -> None:
    """Test conversion of rotation matrices to rotation vectors."""
    rotvecs = rotmat_to_rotvec(rotation_matrices)
    assert torch.allclose(rotvecs, rotation_vectors)


def test_rotquat_to_axis_angle(
    rotation_quaternions: torch.Tensor, rotation_vectors: torch.Tensor, tol: float = 1e-7
) -> None:
    """Test extraction of rotation angles and axes from unit quaternions."""
    rotation_angles = torch.norm(rotation_vectors, dim=-1)
    rotation_axes = torch.where(
        rotation_angles[:, None] < tol,
        torch.zeros_like(rotation_vectors),
        rotation_vectors / rotation_angles[:, None],
    )

    rotangs, rotaxs = _rotquat_to_axis_angle(rotation_quaternions)

    assert torch.allclose(rotangs, rotation_angles)
    assert torch.allclose(rotaxs, rotation_axes)


def test_rotquat_to_rotvec(
    rotation_quaternions: torch.Tensor, rotation_vectors: torch.Tensor
) -> None:
    """Test conversion of unit quaternions to rotation vectors."""
    rotvecs = rotquat_to_rotvec(rotation_quaternions)
    assert torch.allclose(rotvecs, rotation_vectors)


def test_rotquat_to_rotmat(
    rotation_quaternions: torch.Tensor, rotation_matrices: torch.Tensor
) -> None:
    """Test conversion of unit quaternions to rotation matrices."""
    rotmats = rotquat_to_rotmat(rotation_quaternions)
    assert torch.allclose(rotmats, rotation_matrices, atol=1e-5)


def test_apply_rotvec_to_rotmat(rotation_matrices: torch.Tensor, rotation_vectors: torch.Tensor):
    """Test composition of rotation matrix and vector."""
    compound_rots = apply_rotvec_to_rotmat(rotation_matrices, rotation_vectors)
    compound_rotation = torch.einsum(
        "...ij,...jk->...ik", rotation_matrices, rotvec_to_rotmat(rotation_vectors)
    )
    assert torch.allclose(compound_rots, compound_rotation)


def _assert_rotation_matrix(matrices: torch.Tensor, tol: float = 1e-6) -> None:
    """Check if a batch of matrices are valid rotation matrices."""
    # Number of dimensions.
    assert matrices.ndim >= 2

    # 3 x 3 shape for SO(3).
    matrix_dims = matrices.shape[-2:]
    assert matrix_dims == (3, 3)

    # Check inverse equals transpose (U.T U = Id).
    mtm = torch.einsum("b...ij,b...ik->b...jk", matrices, matrices)
    id3 = _broadcast_identity(mtm)
    assert torch.allclose(mtm, id3, atol=tol)

    # Check determinants equal 1.
    assert torch.allclose(
        torch.linalg.det(matrices),
        torch.tensor(1.0, device=matrices.device, dtype=matrices.dtype),
        atol=tol,
    )


def test_scale_rotmat(rotation_matrices, tol=1e-7):
    """Test for scaling rotation matrices by scalars."""
    scales = torch.rand(rotation_matrices.shape[0], 1).to(rotation_matrices)
    angles_0, _, _ = angle_from_rotmat(rotation_matrices)
    scaled_mat = scale_rotmat(rotation_matrices, scales)

    # Scaled matrices should be rotation matrices.
    _assert_rotation_matrix(scaled_mat)

    # Angles should be scaled accordingly.
    angles_1, _, _ = angle_from_rotmat(scaled_mat)
    assert torch.allclose(angles_1 / (angles_0 + tol), scales.squeeze(-1), atol=1e-4)


def test_igso3_derivative(rotation_angles, lower=2e-1, l_max=1000, tol: float = 1e-7):
    """Test implementation of IGSO(3) expansion derivative."""
    # Generate sigma values for testing.
    sigma = torch.clamp(torch.rand(rotation_angles.shape[0]), min=lower, max=0.9)

    # Generate grid for expansions.
    l_grid = torch.arange(l_max + 1)

    # Enable grad for derivatives.
    rotangs = rotation_angles.clone()
    rotangs.requires_grad = True

    # Compute grad using autograd.
    digso3_auto = torch.autograd.grad(
        igso3_expansion(rotangs, sigma, l_grid, tol=tol).sum(), rotangs
    )[0]

    # Compare to manual implementation (tolerance needs to be increased due to numerics).
    digso3 = digso3_expansion(rotangs, sigma, l_grid, tol=tol)

    assert torch.allclose(digso3, digso3_auto, atol=1e-3, rtol=1e-3)


def test_dlog_igso3_derivative(rotation_angles, lower=2e-1, l_max=1000, tol: float = 1e-7):
    """Test derivative of the logarithm of the IGSO(3) expansion."""
    # Generate sigma values for testing.
    sigma = torch.clamp(torch.rand(rotation_angles.shape[0]), min=lower, max=0.9).double()

    # Generate grid for expansions.
    l_grid = torch.arange(l_max + 1).double()

    # Enable grad for derivatives.
    rotangs = rotation_angles.clone().double()
    rotangs.requires_grad = True

    # Compute grad using autograd.
    dlog_igso3_auto = torch.autograd.grad(
        torch.log(torch.abs(igso3_expansion(rotangs, sigma, l_grid, tol=tol)) + tol).sum(), rotangs
    )[0]

    # Compare to manual implementation (tolerance needs to be increased due to numerics).
    dlog_igso3 = dlog_igso3_expansion(rotangs, sigma, l_grid, tol=tol)

    assert torch.allclose(dlog_igso3, dlog_igso3_auto, atol=1e-3, rtol=1e-3)


def test_sample_igso3(
    num_omega: int = NUM_OMEGA,
    num_sigma: int = NUM_SIGMA,
    num_samples: int = NUM_SAMPLES,
    lower: float = 1e-4,
) -> None:
    """
    Test IGSO(3) sampling routine. Only shapes and whether valid rotation matrices are returned are
    checked.
    """

    # Construct sigma grid and samples.
    sigma_grid = torch.linspace(lower, 1.0, num_sigma)
    sigma = torch.clamp(torch.rand(num_sigma), min=lower)

    sampler = SampleIGSO3(
        num_omega=num_omega,
        sigma_grid=sigma_grid,
        l_max=1000,
    )

    # Check if lookup table has correct dimensions
    assert sampler.sigma_grid.shape == (num_sigma,)
    assert sampler.omega_grid.shape == (num_omega,)
    assert sampler.cdf_igso3.shape == (num_sigma, num_omega)

    # Sample and check shape and whether proper rotation matrices were returned.
    samples = sampler.sample(sigma, num_samples)
    assert samples.shape == (num_sigma, num_samples, 3, 3)
    _assert_rotation_matrix(samples)

    # Check sampling of rotation axes for proper shape and unit length.
    samples_vectors = sampler.sample_vector(num_sigma, num_samples)
    samples_vectors_norm = torch.norm(samples_vectors, dim=-1)
    assert samples_vectors.shape == (num_sigma, num_samples, 3)
    assert torch.allclose(samples_vectors_norm, torch.ones_like(samples_vectors_norm))

    # Check shape of sampled angles.
    samples_angles = sampler.sample_angle(sigma, num_samples)
    assert samples_angles.shape == (num_sigma, num_samples)


def test_sample_uso3(
    num_omega: int = NUM_OMEGA,
    num_sigma: int = NUM_SIGMA,
    num_samples: int = NUM_SAMPLES,
    lower: float = 1e-4,
) -> None:
    """
    Test uniform SO(3) sampling routine. Only shapes and whether valid rotation matrices are
    returned are checked.
    """

    # Construct sigma grid and samples.
    sigma_grid = torch.linspace(lower, 1.0, num_sigma)
    sigma = torch.clamp(torch.rand(num_sigma), min=lower)

    sampler = SampleUSO3(
        num_omega=num_omega,
        sigma_grid=sigma_grid,
    )

    # Check if lookup table has correct dimensions
    assert sampler.sigma_grid.shape == (num_sigma,)
    assert sampler.omega_grid.shape == (num_omega,)
    assert sampler.cdf_igso3.shape == (1, num_omega)

    # Sample and check shape and whether proper rotation matrices were returned.
    samples = sampler.sample(sigma, num_samples)
    assert samples.shape == (num_sigma, num_samples, 3, 3)
    _assert_rotation_matrix(samples)

    # Check sampling of rotation axes for proper shape and unit length.
    samples_vectors = sampler.sample_vector(num_sigma, num_samples)
    samples_vectors_norm = torch.norm(samples_vectors, dim=-1)
    assert samples_vectors.shape == (num_sigma, num_samples, 3)
    assert torch.allclose(samples_vectors_norm, torch.ones_like(samples_vectors_norm))

    # Check shape of sampled angles.
    samples_angles = sampler.sample_angle(sigma, num_samples)
    assert samples_angles.shape == (num_sigma, num_samples)

    # Check shape sampling for USO(3).
    samples = sampler.sample_shape(num_sigma, num_samples)
    assert samples.shape == (num_sigma, num_samples, 3, 3)
    _assert_rotation_matrix(samples)


def test_score_so3(
    rotation_vectors,
    num_omega: int = NUM_OMEGA,
    num_sigma: int = NUM_SIGMA,
    num_samples: int = NUM_SAMPLES,
    lower: float = 1e-1,
):
    """
    Test SO(3) score computation. Does not check for overall correctness of score, only for
    tensor shapes and whether derivatives are properly propagated through the score function.
    """
    # Construct sigma grid and samples.
    sigma_grid = torch.linspace(1e-3, 1.5, num_sigma)
    sigma = torch.clamp(torch.rand(num_samples), min=lower)

    score_so3 = ScoreSO3(
        num_omega=num_omega,
        sigma_grid=sigma_grid,
        l_max=2000,
        tol=1e-6,
    )

    # Test computation of score norm (shapes and general sanity only).
    score_scaling = score_so3.get_score_scaling(sigma)
    assert score_scaling.shape == sigma.shape
    assert torch.all(score_scaling >= 0.0)

    # Enable grad for rotation vectors.
    rotvecs = rotation_vectors.clone()
    rotvecs.requires_grad = True

    # Check whether score has the correct shape.
    score = score_so3(sigma, rotvecs)
    assert score.shape == rotvecs.shape
    # Check whether grad has the correct shape.
    dscore = torch.autograd.grad(score.sum(), rotvecs)[0]
    assert dscore.shape == rotvecs.shape
