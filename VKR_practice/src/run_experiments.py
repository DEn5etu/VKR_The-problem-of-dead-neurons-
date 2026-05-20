"""
Experiment runner for the Hopfield dead-neurons practice project.

This is the pre-Russian-labels version: plot labels and titles are in English.
Put this file into:
    src/py_experiments.py
or run it as:
    python -m src.py_experiments --out outputs --grid 61 --n-ic 80
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .hopfield_dead_neurons import (
    bias_sensitivity_scan,
    compare_original_modified,
    find_equilibria_2d,
    flat_energy_scan,
    grid_slr_test_2d,
    integrate_fixed_rk4,
    make_random_demo_model,
    sample_initial_conditions,
    spectral_diagnostics,
)


# -----------------------------------------------------------------------------
# Model creation and reproducibility
# -----------------------------------------------------------------------------


def make_models_for_run(w_seed: Optional[int] = None) -> dict:
    """
    Create random demo models for one experiment run.

    If w_seed is None, a new W is sampled every time.
    If w_seed is fixed, the run is reproducible.
    """
    rng = np.random.default_rng(w_seed)

    relu_seed = int(rng.integers(0, 2**31 - 1))
    softmax_seed = int(rng.integers(0, 2**31 - 1))

    return {
        "relu": make_random_demo_model(
            "relu",
            n=2,
            w_norm=0.65,
            b_scale=0.35,
            symmetric=True,
            seed=relu_seed,
        ),
        "softmax": make_random_demo_model(
            "softmax",
            n=2,
            w_norm=0.65,
            b_scale=0.35,
            symmetric=True,
            seed=softmax_seed,
        ),
    }


def save_model_parameters(out_dir: Path, models: dict) -> None:
    """Save W and b for reproducibility."""
    for name, model in models.items():
        data = {
            "activation": name,
            "W": model.W.tolist(),
            "b": model.b.tolist(),
        }

        path = out_dir / f"model_parameters_{name}.json"

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _dynamics_to_internal_name(dynamics: str) -> str:
    """
    Internally, the model method is named modified_y.
    In figures and output files, we use modified.
    """
    dynamics = dynamics.lower()

    if dynamics == "modified":
        return "modified_y"

    return dynamics


def _get_field(model, dynamics: str):
    """Return a vector field by its public name."""
    dynamics = _dynamics_to_internal_name(dynamics)

    if dynamics == "original":
        return model.original_field

    if dynamics == "modified_y":
        return model.modified_y_field

    raise ValueError(f"Unknown dynamics: {dynamics}")


def _get_energy(model, energy: str):
    """Return an energy function by its public name."""
    if energy == "original":
        return model.original_energy

    if energy in {"E3_y", "E3", "modified"}:
        return model.energy_E3_y

    raise ValueError(f"Unknown energy: {energy}")


def _activation_title(name: str) -> str:
    if name.lower() == "relu":
        return "ReLU"
    if name.lower() == "softmax":
        return "softmax"
    return name


# -----------------------------------------------------------------------------
# Vector fields and energy levels
# -----------------------------------------------------------------------------


def plot_vector_field_and_energy(
    ax_vec,
    ax_energy,
    model,
    dynamics: str,
    energy: str,
    title: str,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    grid_size: int = 25,
) -> None:
    """
    Plot a normalized vector field and energy contours.

    Public dynamics names:
        original
        modified
    """
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)

    X, Y = np.meshgrid(xs, ys)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    E = np.zeros_like(X)

    field = _get_field(model, dynamics)
    energy_fun = _get_energy(model, energy)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            y = np.array([X[i, j], Y[i, j]], dtype=float)

            f = field(0.0, y)
            f_norm = np.linalg.norm(f)

            if f_norm > 1e-12:
                f = f / f_norm

            U[i, j] = f[0]
            V[i, j] = f[1]
            E[i, j] = energy_fun(y)

    ax_vec.quiver(X, Y, U, V, angles="xy")
    ax_vec.set_title(f"Vector field: {title}")
    ax_vec.set_xlabel(r"$y_1$")
    ax_vec.set_ylabel(r"$y_2$")
    ax_vec.set_aspect("equal", adjustable="box")

    if np.isfinite(E).any():
        lo, hi = np.nanpercentile(E, [5, 95])

        if np.isclose(lo, hi):
            lo = np.nanmin(E)
            hi = np.nanmax(E)

        if not np.isclose(lo, hi):
            levels = np.linspace(lo, hi, 12)
            ax_energy.contour(X, Y, E, levels=levels)
        else:
            ax_energy.imshow(
                E,
                origin="lower",
                extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                aspect="equal",
            )

    ax_energy.set_title(f"Energy levels: {title}")
    ax_energy.set_xlabel(r"$y_1$")
    ax_energy.set_ylabel(r"$y_2$")
    ax_energy.set_aspect("equal", adjustable="box")


def make_figure1_replica(out_dir: Path, models: dict) -> Path:
    """
    Plot the figure with vector fields and energy contours for:
    - ReLU original;
    - softmax original;
    - ReLU modified;
    - softmax modified.
    """
    relu = models["relu"]
    softmax = models["softmax"]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(13, 6),
        constrained_layout=True,
    )

    configs = [
        (relu, "original", "original", "ReLU: original energy"),
        (softmax, "original", "original", "softmax: original energy"),
        (relu, "modified", "E3_y", "ReLU: modified energy"),
        (softmax, "modified", "E3_y", "softmax: modified energy"),
    ]

    for col, (model, dynamics, energy, title) in enumerate(configs):
        plot_vector_field_and_energy(
            axes[0, col],
            axes[1, col],
            model,
            dynamics=dynamics,
            energy=energy,
            title=title,
        )

    path = out_dir / "figure1_replica.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    return path


# -----------------------------------------------------------------------------
# Flat energy directions and bias sensitivity
# -----------------------------------------------------------------------------


def make_flatness_plots(out_dir: Path, models: dict) -> None:
    """
    Check flat energy directions for the current models.

    Creates:
        flatness_relu.csv
        flatness_softmax.csv
        flatness_relu.png
        flatness_softmax.png
    """
    configs = {
        "relu": {
            "y0": np.array([-1.0, 1.2], dtype=float),
            "V": np.array([1.0, 0.0], dtype=float),
            "c_values": np.linspace(0.0, -4.0, 81),
            "title": "ReLU flat direction",
        },
        "softmax": {
            "y0": np.array([0.3, -0.7], dtype=float),
            "V": np.array([1.0, 1.0], dtype=float),
            "c_values": np.linspace(-4.0, 4.0, 101),
            "title": "softmax shift direction",
        },
    }

    for act, cfg in configs.items():
        model = models[act]

        df_original = flat_energy_scan(
            model=model,
            y0=cfg["y0"],
            V=cfg["V"],
            c_values=cfg["c_values"],
            energy="original",
        )

        df_modified = flat_energy_scan(
            model=model,
            y0=cfg["y0"],
            V=cfg["V"],
            c_values=cfg["c_values"],
            energy="E3_y",
        )

        df_original["energy_type"] = "original"
        df_modified["energy_type"] = "modified"

        df = pd.concat([df_original, df_modified], ignore_index=True)
        df.to_csv(out_dir / f"flatness_{act}.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        for energy_type, group in df.groupby("energy_type"):
            ax.plot(
                group["c"],
                group["E_minus_E0"],
                marker="o",
                markersize=2,
                linewidth=1.2,
                label=energy_type,
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(cfg["title"])
        ax.set_xlabel(r"$c$")
        ax.set_ylabel(r"$E(y+Vc)-E(y)$")
        ax.legend()

        fig.savefig(out_dir / f"flatness_{act}.png", dpi=180)
        plt.close(fig)


def make_bias_sensitivity_plots(out_dir: Path, models: dict) -> None:
    """
    Check energy sensitivity to bias shifts.

    Creates:
        bias_sensitivity_relu.csv
        bias_sensitivity_relu.png
        bias_sensitivity_softmax.csv
        bias_sensitivity_softmax.png
    """
    configs = {
        "relu": {
            "y0": np.array([-1.0, 1.2], dtype=float),
            "direction": np.array([1.0, 0.0], dtype=float),
            "deltas": np.linspace(-3.0, 3.0, 121),
            "title": "ReLU: bias sensitivity in a dead direction",
        },
        "softmax": {
            "y0": np.array([0.3, -0.7], dtype=float),
            "direction": np.array([1.0, 1.0], dtype=float),
            "deltas": np.linspace(-3.0, 3.0, 121),
            "title": "softmax: bias shift sensitivity",
        },
    }

    for act, cfg in configs.items():
        model = models[act]

        df_original = bias_sensitivity_scan(
            model=model,
            y0=cfg["y0"],
            bias_direction=cfg["direction"],
            deltas=cfg["deltas"],
            energy="original",
        )

        df_modified = bias_sensitivity_scan(
            model=model,
            y0=cfg["y0"],
            bias_direction=cfg["direction"],
            deltas=cfg["deltas"],
            energy="E3_y",
        )

        df_original["energy_type"] = "original"
        df_modified["energy_type"] = "modified"

        df = pd.concat([df_original, df_modified], ignore_index=True)
        df.to_csv(out_dir / f"bias_sensitivity_{act}.csv", index=False)

        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

        for energy_type, group in df.groupby("energy_type"):
            E0 = group["E"].iloc[0]

            ax.plot(
                group["delta_b"],
                group["E"] - E0,
                linewidth=1.5,
                label=energy_type,
            )

        ax.axhline(0.0, linewidth=1.0)
        ax.set_title(cfg["title"])
        ax.set_xlabel(r"$\Delta b$")
        ax.set_ylabel(r"$E_{\Delta b}(y)-E_{\Delta b_0}(y)$")
        ax.legend()

        fig.savefig(out_dir / f"bias_sensitivity_{act}.png", dpi=180)
        plt.close(fig)


# -----------------------------------------------------------------------------
# SLR tables, heatmaps and Smith condition summary
# -----------------------------------------------------------------------------


def make_slr_tables(out_dir: Path, models: dict, grid_size: int) -> None:
    """
    Save SLR diagnostic tables.

    Output filenames:
        slr_grid_relu_original.csv
        slr_grid_relu_modified.csv
        slr_grid_softmax_original.csv
        slr_grid_softmax_modified.csv
    """
    for act, model in models.items():
        for dynamics in ["original", "modified"]:
            df = grid_slr_test_2d(
                model,
                grid_size=grid_size,
                dynamics=dynamics,
            )

            df.to_csv(
                out_dir / f"slr_grid_{act}_{dynamics}.csv",
                index=False,
            )


def make_slr_heatmap(out_dir: Path) -> None:
    """Plot SLR heatmaps."""
    configs = [
        ("relu", "original"),
        ("relu", "modified"),
        ("softmax", "original"),
        ("softmax", "modified"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)

    for ax, (act, dynamics) in zip(axes.ravel(), configs):
        csv_path = out_dir / f"slr_grid_{act}_{dynamics}.csv"

        if not csv_path.exists():
            ax.set_title(f"{act}, {dynamics}\nCSV not found")
            ax.axis("off")
            continue

        df = pd.read_csv(csv_path)

        pivot = df.pivot(
            index="x2",
            columns="x1",
            values="lambda1_plus_lambda2",
        )

        x_values = pivot.columns.to_numpy(dtype=float)
        y_values = pivot.index.to_numpy(dtype=float)
        Z = pivot.to_numpy(dtype=float)
        Z_masked = np.ma.masked_invalid(Z)

        im = ax.imshow(
            Z_masked,
            origin="lower",
            extent=[
                float(x_values.min()),
                float(x_values.max()),
                float(y_values.min()),
                float(y_values.max()),
            ],
            aspect="equal",
        )

        ax.set_title(f"{_activation_title(act)}, {dynamics}")
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")

        if np.isfinite(Z).any():
            try:
                ax.contour(
                    x_values,
                    y_values,
                    Z,
                    levels=[0.0],
                    linewidths=1.0,
                )
            except Exception:
                pass

        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle(
        r"SLR diagnostic: $\lambda_1(J_{a,s})+\lambda_2(J_{a,s})$",
        fontsize=12,
    )

    fig.savefig(out_dir / "slr_heatmap.png", dpi=180)
    plt.close(fig)


def make_smith_condition_summary(
    out_dir: Path,
    eta: float = 0.0,
) -> pd.DataFrame:
    """
    Count grid points where the Smith condition holds.

    Smith condition:
        lambda1_plus_lambda2 < -eta.
    """
    configs = [
        ("relu", "original"),
        ("relu", "modified"),
        ("softmax", "original"),
        ("softmax", "modified"),
    ]

    rows = []

    for activation, dynamics in configs:
        csv_path = out_dir / f"slr_grid_{activation}_{dynamics}.csv"

        if not csv_path.exists():
            rows.append(
                {
                    "activation": activation,
                    "dynamics": dynamics,
                    "csv_found": False,
                    "total_grid_points": 0,
                    "finite_points": 0,
                    "applicable_points": 0,
                    "smith_true_points": 0,
                    "smith_false_points": 0,
                    "fraction_among_all": np.nan,
                    "fraction_among_finite": np.nan,
                    "fraction_among_applicable": np.nan,
                    "eta": eta,
                }
            )
            continue

        df = pd.read_csv(csv_path)

        total_points = int(len(df))
        values = df["lambda1_plus_lambda2"].to_numpy(dtype=float)
        finite_mask = np.isfinite(values)

        if "smith_applicable" in df.columns:
            applicable_mask = df["smith_applicable"].astype(bool).to_numpy()
        else:
            applicable_mask = df["active_dim"].to_numpy(dtype=int) >= 2

        smith_true_mask = finite_mask & applicable_mask & (values < -eta)

        finite_points = int(np.sum(finite_mask))
        applicable_points = int(np.sum(finite_mask & applicable_mask))
        smith_true_points = int(np.sum(smith_true_mask))
        smith_false_points = int(applicable_points - smith_true_points)

        fraction_among_all = smith_true_points / total_points if total_points > 0 else np.nan
        fraction_among_finite = smith_true_points / finite_points if finite_points > 0 else np.nan
        fraction_among_applicable = (
            smith_true_points / applicable_points if applicable_points > 0 else np.nan
        )

        rows.append(
            {
                "activation": activation,
                "dynamics": dynamics,
                "csv_found": True,
                "total_grid_points": total_points,
                "finite_points": finite_points,
                "applicable_points": applicable_points,
                "smith_true_points": smith_true_points,
                "smith_false_points": smith_false_points,
                "fraction_among_all": fraction_among_all,
                "fraction_among_finite": fraction_among_finite,
                "fraction_among_applicable": fraction_among_applicable,
                "eta": eta,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "smith_condition_summary.csv", index=False)

    with open(out_dir / "smith_condition_summary.txt", "w", encoding="utf-8") as f:
        f.write("Smith condition summary\n")
        f.write("=======================\n\n")
        f.write(f"Condition: lambda1_plus_lambda2 < {-eta}\n\n")

        for _, row in summary.iterrows():
            f.write(f"Activation: {row['activation']}\n")
            f.write(f"Dynamics:   {row['dynamics']}\n")

            if not bool(row["csv_found"]):
                f.write("CSV file:   not found\n\n")
                continue

            f.write(f"Total grid points:       {int(row['total_grid_points'])}\n")
            f.write(f"Finite points:           {int(row['finite_points'])}\n")
            f.write(f"Applicable points:       {int(row['applicable_points'])}\n")
            f.write(f"Smith true points:       {int(row['smith_true_points'])}\n")
            f.write(f"Smith false points:      {int(row['smith_false_points'])}\n")
            f.write(
                "Fraction among all:      "
                f"{100.0 * row['fraction_among_all']:.2f}%\n"
            )
            f.write(
                "Fraction among finite:   "
                f"{100.0 * row['fraction_among_finite']:.2f}%\n"
            )

            if np.isfinite(row["fraction_among_applicable"]):
                f.write(
                    "Fraction among applicable: "
                    f"{100.0 * row['fraction_among_applicable']:.2f}%\n"
                )
            else:
                f.write("Fraction among applicable: undefined\n")

            f.write("\n")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)

    labels = (
        summary["activation"].astype(str)
        + "\n"
        + summary["dynamics"].astype(str)
    ).tolist()

    x = np.arange(len(summary))
    ax.bar(x, summary["fraction_among_all"].to_numpy(dtype=float))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction among all grid points")
    ax.set_title("Grid points satisfying Smith condition")

    for i, row in summary.iterrows():
        value = row["fraction_among_all"]

        if np.isfinite(value):
            ax.text(
                i,
                value + 0.03,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.savefig(out_dir / "smith_condition_fraction.png", dpi=180)
    plt.close(fig)

    return summary


# -----------------------------------------------------------------------------
# Spectral diagnostics
# -----------------------------------------------------------------------------


def make_spectral_reports(out_dir: Path, models: dict) -> None:
    """Save spectral diagnostics for the current models."""
    representative_points = {
        "relu": [
            np.array([-1.0, 1.2], dtype=float),
            np.array([1.0, 1.0], dtype=float),
            np.array([-1.0, -1.0], dtype=float),
        ],
        "softmax": [
            np.array([0.3, -0.7], dtype=float),
            np.array([2.0, -1.0], dtype=float),
            np.array([0.0, 0.0], dtype=float),
        ],
    }

    for act, model in models.items():
        rows = []
        json_data = []

        for point_id, y in enumerate(representative_points[act]):
            diag = spectral_diagnostics(model, y)

            eig_active = np.asarray(diag["eig_Wperp_Lambdaperp_minus_I"])
            eig_energy = np.asarray(diag["eig_Lambda_minus_Lambda_W_Lambda"])
            eig_energy_neg = np.asarray(diag["eig_Lambda_W_Lambda_minus_Lambda"])

            row = {
                "point_id": point_id,
                "y1": float(y[0]),
                "y2": float(y[1]),
                "rank_Lambda": int(diag["rank_Lambda"]),
                "nullity_Lambda": int(diag["nullity_Lambda"]),
                "max_real_eig_active": (
                    float(np.max(np.real(eig_active))) if eig_active.size else np.nan
                ),
                "min_eig_energy_hessian": (
                    float(np.min(eig_energy)) if eig_energy.size else np.nan
                ),
                "max_eig_energy_negative": (
                    float(np.max(eig_energy_neg)) if eig_energy_neg.size else np.nan
                ),
            }

            rows.append(row)

            json_data.append(
                {
                    "point_id": point_id,
                    "y": y.tolist(),
                    "rank_Lambda": int(diag["rank_Lambda"]),
                    "nullity_Lambda": int(diag["nullity_Lambda"]),
                    "Lambda_eigenvalues": np.asarray(
                        diag["Lambda_eigenvalues"]
                    ).tolist(),
                    "eig_Wperp_Lambdaperp_minus_I": [
                        [float(np.real(z)), float(np.imag(z))]
                        for z in eig_active
                    ],
                    "eig_Lambda_minus_Lambda_W_Lambda": eig_energy.tolist(),
                    "eig_Lambda_W_Lambda_minus_Lambda": eig_energy_neg.tolist(),
                }
            )

        pd.DataFrame(rows).to_csv(out_dir / f"spectral_{act}.csv", index=False)

        with open(out_dir / f"spectral_{act}.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)


# -----------------------------------------------------------------------------
# Dynamics comparison and equilibria plots
# -----------------------------------------------------------------------------


def add_equilibria_to_axis(ax, equilibria_df: pd.DataFrame) -> None:
    """
    Add equilibrium points to a trajectory plot.

    Red star: equilibrium of the original system.
    Black cross: modified-only equilibrium.
    """
    if equilibria_df.empty:
        ax.text(
            0.02,
            0.98,
            "equilibria not found",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        return

    original_eq = equilibria_df[equilibria_df["is_original_equilibrium"]]
    modified_only = equilibria_df[
        (~equilibria_df["is_original_equilibrium"])
        & (equilibria_df["is_modified_equilibrium"])
    ]

    if not original_eq.empty:
        ax.scatter(
            original_eq["y1"],
            original_eq["y2"],
            marker="*",
            s=180,
            c="red",
            edgecolors="black",
            linewidths=0.7,
            label="equilibria",
            zorder=5,
        )

        for _, row in original_eq.iterrows():
            ax.text(
                row["y1"],
                row["y2"],
                f" E{int(row['eq_id'])}",
                fontsize=8,
                zorder=6,
            )

    if not modified_only.empty:
        ax.scatter(
            modified_only["y1"],
            modified_only["y2"],
            marker="x",
            s=90,
            c="black",
            label="modified-only eq.",
            zorder=5,
        )


def make_equilibria_text(equilibria_df: pd.DataFrame, max_rows: int = 5) -> str:
    """Create a short text block with equilibrium coordinates."""
    if equilibria_df.empty:
        return "Equilibria: not found"

    original_eq = equilibria_df[equilibria_df["is_original_equilibrium"]]

    if original_eq.empty:
        return "Original equilibria: not found"

    lines = ["Original equilibria:"]

    for _, row in original_eq.head(max_rows).iterrows():
        lines.append(f"E{int(row['eq_id'])}=({row['y1']:.3f}, {row['y2']:.3f})")

    if len(original_eq) > max_rows:
        lines.append(f"... total: {len(original_eq)}")

    return "\n".join(lines)


def make_dynamics_comparison(
    out_dir: Path,
    models: dict,
    n_ic: int,
    seed: int,
) -> None:
    """
    Compare original and modified dynamics.

    Equilibria are found for the current random W and shown on trajectory plots.
    """
    summaries = []

    for act, model in models.items():
        init = sample_initial_conditions(n_ic, model.n, seed=seed)

        equilibria_original = find_equilibria_2d(
            model,
            dynamics="original",
            xlim=(-5.0, 5.0),
            ylim=(-5.0, 5.0),
            grid_size=15,
        )

        equilibria_modified = find_equilibria_2d(
            model,
            dynamics="modified",
            xlim=(-5.0, 5.0),
            ylim=(-5.0, 5.0),
            grid_size=15,
        )

        equilibria_all = pd.concat(
            [equilibria_original, equilibria_modified],
            ignore_index=True,
        )

        equilibria_all = equilibria_all.drop_duplicates(
            subset=["y1", "y2"],
            keep="first",
        )

        equilibria_all.to_csv(out_dir / f"equilibria_{act}.csv", index=False)

        df = compare_original_modified(
            model,
            init,
            t_final=20.0,
            residual_tol=1e-4,
            equilibrium_tol=5e-3,
            dt=0.03,
            equilibria_df=equilibria_original,
        )

        df.to_csv(out_dir / f"compare_dynamics_{act}.csv", index=False)

        summary = (
            df.groupby("dynamics")
            .agg(
                success_rate=("success", "mean"),
                residual_success_rate=("residual_success", "mean"),
                distance_success_rate=("distance_success", "mean"),
                mean_residual=("residual_norm", "mean"),
                median_residual=("residual_norm", "median"),
                mean_eq_distance=("nearest_eq_distance", "mean"),
                median_eq_distance=("nearest_eq_distance", "median"),
                mean_energy_drop=("energy_drop", "mean"),
            )
            .reset_index()
        )

        summary["activation"] = act
        summaries.append(summary)

        trajectory_configs = [
            ("original", model.original_field),
            ("modified", model.modified_y_field),
        ]

        for dynamics_name, field in trajectory_configs:
            fig, ax = plt.subplots(figsize=(7, 6))

            for y0 in init[: min(16, len(init))]:
                result = integrate_fixed_rk4(
                    field=field,
                    y0=y0,
                    t_final=20.0,
                    dt=0.03,
                    save_trajectory=True,
                )

                y_values = np.asarray(result["y_values"])

                ax.plot(
                    y_values[:, 0],
                    y_values[:, 1],
                    linewidth=1.0,
                    alpha=0.8,
                )

                ax.scatter(
                    y_values[0, 0],
                    y_values[0, 1],
                    s=12,
                    marker="o",
                    alpha=0.7,
                )

                ax.scatter(
                    y_values[-1, 0],
                    y_values[-1, 1],
                    s=18,
                    marker="x",
                    alpha=0.9,
                )

            add_equilibria_to_axis(ax, equilibria_all)

            ax.set_title(f"{dynamics_name} trajectories: {_activation_title(act)}")
            ax.set_xlabel(r"$y_1$")
            ax.set_ylabel(r"$y_2$")
            ax.set_aspect("equal", adjustable="box")

            eq_text = make_equilibria_text(equilibria_all)

            ax.text(
                0.02,
                0.02,
                eq_text,
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

            ax.legend(loc="upper right", fontsize=8)

            fig.savefig(out_dir / f"trajectories_{dynamics_name}_{act}.png", dpi=180)
            plt.close(fig)

    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(out_dir / "summary_dynamics_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    labels = (all_summary["activation"] + "\n" + all_summary["dynamics"]).tolist()
    x = np.arange(len(all_summary))

    axes[0].bar(x, all_summary["success_rate"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("success rate")
    axes[0].set_title("Successful convergence")

    axes[1].bar(x, all_summary["median_residual"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("median residual norm")
    axes[1].set_title("Final original residual")

    fig.savefig(out_dir / "dynamics_comparison.png", dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=str,
        default="outputs",
        help="Output directory.",
    )

    parser.add_argument(
        "--grid",
        type=int,
        default=61,
        help="Grid size for the SLR test.",
    )

    parser.add_argument(
        "--n-ic",
        type=int,
        default=80,
        help="Number of initial conditions.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for initial conditions.",
    )

    parser.add_argument(
        "--w-seed",
        type=int,
        default=None,
        help="Seed for random W. If omitted, W is resampled every run.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = make_models_for_run(w_seed=args.w_seed)
    save_model_parameters(out_dir, models)

    make_figure1_replica(out_dir, models)
    make_flatness_plots(out_dir, models)
    make_bias_sensitivity_plots(out_dir, models)

    make_slr_tables(out_dir, models, grid_size=args.grid)
    make_smith_condition_summary(out_dir, eta=0.0)
    make_slr_heatmap(out_dir)

    make_spectral_reports(out_dir, models)
    make_dynamics_comparison(out_dir, models, n_ic=args.n_ic, seed=args.seed)

    print(f"Done. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
