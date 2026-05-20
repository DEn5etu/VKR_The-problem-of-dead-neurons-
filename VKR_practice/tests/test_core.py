"""Минимальные unit-тесты для практической части ВКР."""

import numpy as np

from src.hopfield_dead_neurons import (
    flat_energy_scan,
    make_demo_model,
    orthogonal_projectors_from_lambda,
    smith_truncated_trace,
)


def test_relu_original_energy_flat_in_dead_direction():
    model = make_demo_model("relu")
    y0 = np.array([-1.0, 1.2])
    V = np.array([1.0, 0.0])
    c_values = np.linspace(0.0, -4.0, 20)
    df = flat_energy_scan(model, y0, V, c_values, energy="original")
    assert np.max(np.abs(df["E_minus_E0"].to_numpy())) < 1e-10


def test_softmax_original_energy_shift_invariant():
    model = make_demo_model("softmax")
    y0 = np.array([0.3, -0.7])
    V = np.array([1.0, 1.0])
    c_values = np.linspace(-4.0, 4.0, 20)
    df = flat_energy_scan(model, y0, V, c_values, energy="original")
    assert np.max(np.abs(df["E_minus_E0"].to_numpy())) < 1e-10


def test_projectors_are_orthogonal_and_sum_to_identity():
    model = make_demo_model("softmax")
    Lam = model.Lambda(np.array([0.1, -0.2]))
    info = orthogonal_projectors_from_lambda(Lam)
    Pa = info["Pa"]
    Pd = info["Pd"]
    I = np.eye(2)
    assert np.linalg.norm(Pa @ Pd) < 1e-10
    assert np.linalg.norm(Pa + Pd - I) < 1e-10


def test_smith_truncated_trace_two_dimensional_case():
    J = np.array([[-2.0, 1.0], [0.0, -3.0]])
    result = smith_truncated_trace(J)
    assert result["smith_applicable"] is True
    assert result["lambda1_plus_lambda2"] < 0.0
