"""
hopfield_dead_neurons.py
=========================

Практическая часть ВКР:
"Проблема мёртвых нейронов и плоских областей функционала энергии
в современной сети Хопфилда: динамико-системный анализ".

Файл содержит реализацию численных объектов из plan.pdf и замечаний VKR:
1. исходная динамика Krotov--Hopfield
       y_dot = W g(y) - y + b;
2. исходная энергия
       E(y) = (y-b)^T g(y) - L(y) - 1/2 g(y)^T W g(y);
3. модифицированная динамика из статьи Fanaskov--Oseledets в двух вариантах:
       u_dot = R(u) [g(Wu+b) - u],
   и удобный для репликации Figure 1 вариант
       y_dot = R(y) [Wg(y) - y + b],  R(y)=I-W Lambda(y);
4. вычисление Lambda(y)=Dg(y)=nabla^2 L(y), Range Lambda, Ker Lambda,
   проекторов Pa, Pd;
5. численная проверка SLR-критерия Смита:
       lambda_1(J_s^(a)) + lambda_2(J_s^(a)) < 0;
6. тесты плоских направлений E(y+Vc)=E(y), спектральные диагностики,
   интегрирование ОДУ и сравнение исходной/модифицированной динамики.

Код намеренно написан на NumPy/SciPy, а не на PyTorch/JAX: так легче проверить
формулы, спектры и получить полностью воспроизводимые графики для отчёта.
При необходимости этот файл можно перенести на JAX почти построчно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh, norm
from scipy.integrate import solve_ivp
from scipy.special import erf, expit, logsumexp

from typing import Callable, Dict, Optional
from scipy.optimize import root
from numpy.linalg import norm


def _normalize_spectral_norm(W: Array, target_norm: float = 0.65) -> Array:
    """
    Масштабирует матрицу так, чтобы её спектральная норма была около target_norm.

    Это нужно, чтобы случайная система не была слишком жёсткой и чтобы
    траектории чаще сходились к равновесиям.
    """
    W = np.asarray(W, dtype=float)

    current_norm = norm(W, 2)
    if current_norm < 1e-12:
        return W

    return W * (target_norm / current_norm)





Array = np.ndarray


# -----------------------------------------------------------------------------
# 1. Функции активации g, потенциал L и матрица Lambda = Dg = Hess L
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Activation:
    """Контейнер для функции активации g = grad L и её гессиана Lambda.

    name:
        Человеческое имя активации.
    g:
        Векторная функция g(y).
    L:
        Скалярный потенциал L(y), такой что grad L = g.
        Для некоторых экспериментальных гладких активаций L может быть не задан.
    Lambda:
        Якобиан Dg(y), в обозначениях ВКР это Lambda(y)=nabla^2 L(y).
    """

    name: str
    g: Callable[[Array], Array]
    L: Callable[[Array], float]
    Lambda: Callable[[Array], Array]


class MissingPotentialError(NotImplementedError):
    """Ошибка для активаций, где L не задан в закрытой форме."""


def stable_softmax(y: Array) -> Array:
    """Численно устойчивая softmax(y)."""
    y = np.asarray(y, dtype=float)
    z = y - np.max(y)
    ez = np.exp(z)
    return ez / np.sum(ez)


def relu_activation() -> Activation:
    """ReLU: g_i(y)=max(y_i,0), L=1/2 sum ReLU(y_i)^2.

    В точке y_i=0 производная ReLU не определена. Для численных экспериментов
    берём правдоподобный подградиент 0. Это удобно для выявления мёртвых
    направлений: y_i <= 0 даёт Lambda_ii=0.
    """

    def g(y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        return np.maximum(y, 0.0)

    def L(y: Array) -> float:
        r = g(y)
        return 0.5 * float(np.dot(r, r))

    def Lambda(y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        return np.diag((y > 0.0).astype(float))

    return Activation("ReLU", g, L, Lambda)


def sigmoid_activation() -> Activation:
    """Сигмоида: g_i(y)=1/(1+exp(-y_i)), L=sum log(1+exp(y_i))."""

    def g(y: Array) -> Array:
        return expit(np.asarray(y, dtype=float))

    def L(y: Array) -> float:
        y = np.asarray(y, dtype=float)
        # logaddexp(0,y) = log(1+exp(y)) без переполнения.
        return float(np.sum(np.logaddexp(0.0, y)))

    def Lambda(y: Array) -> Array:
        s = g(y)
        return np.diag(s * (1.0 - s))

    return Activation("sigmoid", g, L, Lambda)


def softmax_activation() -> Activation:
    """Softmax: g(y)=softmax(y), L(y)=log sum_i exp(y_i).

    Lambda = diag(s) - s s^T всегда имеет ядро, содержащее вектор единиц.
    Поэтому энергия исходной модели инвариантна к сдвигу y -> y + c*1.
    """

    def g(y: Array) -> Array:
        return stable_softmax(y)

    def L(y: Array) -> float:
        return float(logsumexp(np.asarray(y, dtype=float)))

    def Lambda(y: Array) -> Array:
        s = g(y)
        return np.diag(s) - np.outer(s, s)

    return Activation("softmax", g, L, Lambda)


def smooth_leaky_relu_activation() -> Activation:
    """Гладкая Leaky-ReLU/SMU-активация из примера Fanaskov--Oseledets.

    g(u) = u * (5 + 3 erf(3u/8)) / 8.

    Потенциал L для этой функции в коде не нужен для E2/E3, поэтому L не задан.
    Если потребуется исходная энергия E(y), лучше добавить аналитический или
    численный интеграл от g.
    """

    def g(y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        return y * (5.0 + 3.0 * erf(3.0 * y / 8.0)) / 8.0

    def L(y: Array) -> float:
        raise MissingPotentialError(
            "Для smooth_leaky_relu в этом прототипе не задан потенциал L. "
            "Используйте энергии E2/E3, где L не требуется."
        )

    def Lambda(y: Array) -> Array:
        y = np.asarray(y, dtype=float)
        a = (5.0 + 3.0 * erf(3.0 * y / 8.0)) / 8.0
        da = 9.0 * y * np.exp(-9.0 * y * y / 64.0) / (32.0 * np.sqrt(np.pi))
        return np.diag(a + da)

    return Activation("smooth_leaky_relu", g, L, Lambda)


def get_activation(name: str) -> Activation:
    """Фабрика активаций по имени."""
    key = name.lower().replace("-", "_")
    if key == "relu":
        return relu_activation()
    if key == "sigmoid":
        return sigmoid_activation()
    if key == "softmax":
        return softmax_activation()
    if key in {"smooth_leaky_relu", "smooth_relu", "smu"}:
        return smooth_leaky_relu_activation()
    raise ValueError(f"Неизвестная активация: {name!r}")


# -----------------------------------------------------------------------------
# 2. Базовая модель Krotov--Hopfield и модифицированные динамики
# -----------------------------------------------------------------------------


@dataclass
class HopfieldModel:

    """
    Модель современной сети Хопфилда/ассоциативной памяти.

    W:
        Матрица весов размера n x n. Для строгой исходной энергии обычно W=W^T,
        но код не запрещает несимметричные W для численных экспериментов.
    b:
        Вектор сдвига размера n.
    activation:
        Функция активации g=grad L и Lambda=Dg.
    """

    W: Array
    b: Array
    activation: Activation

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        self.b = np.asarray(self.b, dtype=float)
        if self.W.ndim != 2 or self.W.shape[0] != self.W.shape[1]:
            raise ValueError("W должна быть квадратной матрицей.")
        if self.b.shape != (self.W.shape[0],):
            raise ValueError("b должен иметь форму (n,), где n=W.shape[0].")

    @property
    def n(self) -> int:
        return int(self.W.shape[0])

    def g(self, y: Array) -> Array:
        return self.activation.g(np.asarray(y, dtype=float))

    def L(self, y: Array) -> float:
        return self.activation.L(np.asarray(y, dtype=float))

    def Lambda(self, y: Array) -> Array:
        return self.activation.Lambda(np.asarray(y, dtype=float))

    # ----- исходная модель y_dot = W g(y) - y + b -----

    def original_residual(self, y: Array) -> Array:
        """
        Правая часть исходной системы F(y)=Wg(y)-y+b.
        """
        y = np.asarray(y, dtype=float)
        return self.W @ self.g(y) - y + self.b

    def original_field(self, t: float, y: Array) -> Array:
        """
        Сигнатура для solve_ivp: y_dot = F(y).
        """
        return self.original_residual(y)

    def original_jacobian(self, y: Array) -> Array:
        """
        Якобиан исходной динамики: J(y)=W Lambda(y)-I.
        """
        return self.W @ self.Lambda(y) - np.eye(self.n)

    def original_energy(self, y: Array) -> float:
        """
        Исходная функция энергии Krotov--Hopfield.

        E(y) = (y-b)^T g(y) - L(y) - 1/2 g(y)^T W g(y).
        Именно эта энергия имеет плоские направления при мёртвых нейронах.
        """
        y = np.asarray(y, dtype=float)
        gy = self.g(y)
        return float((y - self.b) @ gy - self.L(y) - 0.5 * gy @ self.W @ gy)

    # ----- модифицированная y-динамика из уравнений (13)--(14) статьи -----
    def R_y_default(self, y: Array) -> Array:
        """
        Матрица R(y)=I -W Lambda(y) для модифицированной y-динамики.

        Этот вариант используется в статье для рисунка, аналогичного Figure 1:
            y_dot = R(y) [Wg(y)-y+b],
            E3(y) = 1/2 ||Wg(y)-y+b||^2.
        """
        return np.eye(self.n) - self.W @ self.Lambda(y)

    def modified_y_field(self, t: float, y: Array) -> Array:
        """
        Модифицированная y-динамика без плоской энергии E3.
        """
        return self.R_y_default(y) @ self.original_residual(y)

    def energy_E3_y(self, y: Array, S: Optional[Array] = None) -> float:
        """
        Энергия E3(y)=1/2 residual^T S residual для modified_y_field.

        По умолчанию S=I. В отличие от исходной энергии, эта энергия зависит
        от полного y через residual=Wg(y)-y+b и обычно не имеет плоских листов.
        """
        r = self.original_residual(y)
        if S is None:
            return 0.5 * float(r @ r)
        S = np.asarray(S, dtype=float)
        return 0.5 * float(r @ S @ r)

    # ----- модифицированная u-динамика u_dot = R(u)[g(Wu+b)-u] -----

    def u_residual(self, u: Array) -> Array:
        """
        f(u)=g(Wu+b)-u для модифицированной u-системы.
        """
        u = np.asarray(u, dtype=float)
        return self.g(self.W @ u + self.b) - u

    def R_u_default(self, u: Array) -> Array:
        """
        Простой выбор R(u)=I-W^T Lambda(Wu+b) для E2.

        Этот выбор соответствует практическому варианту из статьи для гладких
        активаций и несимметричных W. В ReLU-точках с разрывной производной
        его нужно интерпретировать как численный прототип.
        """
        z = self.W @ np.asarray(u, dtype=float) + self.b
        return np.eye(self.n) - self.W.T @ self.Lambda(z)

    def modified_u_field(self, t: float, u: Array) -> Array:
        """
        u_dot = R(u) [g(Wu+b)-u].
        """
        return self.R_u_default(u) @ self.u_residual(u)

    def energy_E2_u(self, u: Array) -> float:
        """
        E2(u)=1/2 ||g(Wu+b)-u||^2 для u-динамики.
        """
        f = self.u_residual(u)
        return 0.5 * float(f @ f)


# -----------------------------------------------------------------------------
# 3. Линейная алгебра: Range Lambda, Ker Lambda, проекторы Pa/Pd
# -----------------------------------------------------------------------------

def symmetrize(A: Array) -> Array:
    """
    Симметризированная часть матрицы.
    """
    A = np.asarray(A, dtype=float)
    return 0.5 * (A + A.T)

def orthogonal_projectors_from_lambda(Lambda: Array, tol: float = 1e-9) \
        -> Dict[str, Array | int | float]:
    """
    Вычислить Range Lambda, Ker Lambda и проекторы Pa/Pd.

    Для симметричной Lambda берём спектральное разложение. Активное
    подпространство Range Lambda задаётся собственными векторами с |lambda|>tol,
    а плоское подпространство энергии Ker Lambda — остальными собственными
    векторами.
    """

    Lam = symmetrize(Lambda)
    vals, vecs = np.linalg.eigh(Lam)
    active_mask = np.abs(vals) > tol
    Qa = vecs[:, active_mask]
    Qd = vecs[:, ~active_mask]
    n = Lam.shape[0]
    Pa = Qa @ Qa.T if Qa.size else np.zeros((n, n))
    Pd = Qd @ Qd.T if Qd.size else np.zeros((n, n))
    return {
        "eigenvalues": vals,
        "rank": int(np.sum(active_mask)),
        "nullity": int(np.sum(~active_mask)),
        "Qa": Qa,
        "Qd": Qd,
        "Pa": Pa,
        "Pd": Pd,
        "tol": float(tol),
    }


def restrict_to_basis(A: Array, Q: Array) -> Array:
    """
    Матрица линейного оператора A в ортонормированном базисе Q.
    """
    if Q.size == 0:
        return np.zeros((0, 0))
    return Q.T @ A @ Q


def recover_steady_from_energy_minimum(model: HopfieldModel, y_energy_min: Array) -> Array:
    """
    Восстановление полной стационарной точки по формуле из Proposition 4.

    Если оптимизация исходной энергии нашла точку y_energy_min, то из-за
    плоских направлений она может не быть стационарной точкой полной ОДУ.
    Нужно добавить проекцию на Ker Lambda:
        y_* = y_energy_min + Pd [W g(y_energy_min) + b].
    """
    info = orthogonal_projectors_from_lambda(model.Lambda(y_energy_min))
    Pd = info["Pd"]
    return np.asarray(y_energy_min, dtype=float) + Pd @ (model.W @ model.g(y_energy_min) + model.b)


# -----------------------------------------------------------------------------
# 4. SLR/Smith-диагностики: укороченная сумма собственных значений
# -----------------------------------------------------------------------------


def smith_truncated_trace(J_active: Array) -> Dict[str, Array | float | int | bool]:
    """
    Вычислить lambda_1 + lambda_2 для симметризованного активного якобиана.

    В теореме Смита для активной подсистемы проверяется знак суммы двух
    наибольших собственных значений J_s^(a). Если активная размерность меньше 2,
    полноценный критерий отсутствия циклов не применяется; в таком случае
    возвращаем размерность и доступное собственное значение для диагностики.
    """
    d = int(J_active.shape[0])
    if d == 0:
        return {
            "active_dim": 0,
            "eigvals_desc": np.array([]),
            "lambda1_plus_lambda2": np.nan,
            "smith_applicable": False,
        }
    eigs = np.sort(eigvalsh(symmetrize(J_active)))[::-1]
    if d == 1:
        return {
            "active_dim": 1,
            "eigvals_desc": eigs,
            "lambda1_plus_lambda2": float(eigs[0]),
            "smith_applicable": False,
        }
    return {
        "active_dim": d,
        "eigvals_desc": eigs,
        "lambda1_plus_lambda2": float(eigs[0] + eigs[1]),
        "smith_applicable": True,
    }

def slr_diagnostic_at_point(
    model: HopfieldModel,
    y: Array,
    jacobian: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-9,
) -> Dict[str, object]:
    """
    SLR-диагностика в точке y.

    По замечаниям к ВКР корректнее проверять SLR на активной подсистеме.
    Численно это реализовано как ограничение якобиана на Range Lambda(y):
        J_a = Qa^T J(y) Qa,
    где столбцы Qa образуют ортонормированный базис Range Lambda(y).
    """
    y = np.asarray(y, dtype=float)
    Lam = model.Lambda(y)
    info = orthogonal_projectors_from_lambda(Lam, tol=tol)
    Qa = info["Qa"]
    J = model.original_jacobian(y) if jacobian is None else jacobian(y)
    Ja = restrict_to_basis(J, Qa)
    smith = smith_truncated_trace(Ja)
    return {
        "y": y,
        "rank_Lambda": info["rank"],
        "nullity_Lambda": info["nullity"],
        "Lambda_eigenvalues": info["eigenvalues"],
        "J_active": Ja,
        "smith": smith,
        "Pa": info["Pa"],
        "Pd": info["Pd"],
    }


def numerical_jacobian(
    f: Callable[[Array], Array],
    x: Array,
    eps: float = 1e-6
) -> Array:
    """
    Центральная разность для якобиана произвольного векторного поля.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    J = np.zeros((n, n), dtype=float)
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = eps
        J[:, j] = (f(x + dx) - f(x - dx)) / (2.0 * eps)
    return J

def grid_slr_test_2d(
    model: HopfieldModel,
    xlim=(-3.0, 3.0),
    ylim=(-3.0, 3.0),
    grid_size: int = 61,
    dynamics: str = "original",
    tol: float = 1e-9,
) -> pd.DataFrame:
    if model.n != 2:
        raise ValueError("grid_slr_test_2d is only for n=2.")

    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)

    rows = []

    dynamics_key = dynamics.lower()

    if dynamics_key == "original":
        jac = model.original_jacobian
    elif dynamics_key in {"modified", "modified_y"}:
        jac = lambda z: numerical_jacobian(
            lambda q: model.modified_y_field(0.0, q), z
        )
    else:
        raise ValueError("Unknown dynamics.")

    for x1 in xs:
        for x2 in ys:
            y = np.array([x1, x2], dtype=float)

            diag = slr_diagnostic_at_point(
                model,
                y,
                jacobian=jac,
                tol=tol,
            )

            smith = diag["smith"]

            rows.append(
                {
                    "x1": x1,
                    "x2": x2,
                    "rank_Lambda": int(diag["rank_Lambda"]),
                    "nullity_Lambda": int(diag["nullity_Lambda"]),
                    "active_dim": int(smith["active_dim"]),
                    "smith_applicable": bool(smith["smith_applicable"]),
                    "lambda1_plus_lambda2": float(
                        smith["lambda1_plus_lambda2"]
                    ),
                }
            )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. Проверка плоских направлений и спектры
# -----------------------------------------------------------------------------


def flat_energy_scan(
    model: HopfieldModel,
    y0: Array,
    V: Array,
    c_values: Iterable[float],
    energy: str = "original",
) -> pd.DataFrame:
    """
    Проверить E(y0 + V c)=E(y0) на сетке значений c.

    'original' - исходная Krotov--Hopfield энергия;
    'E3_y' - модифицированная энергия 1/2||Wg(y)-y+b||^2.
    """
    y0 = np.asarray(y0, dtype=float)
    V = np.asarray(V, dtype=float)
    if V.ndim == 1:
        V = V.reshape(-1, 1)
    if V.shape[0] != y0.size:
        raise ValueError("V должен иметь форму (n,k) или (n,).")

    if energy == "original":
        energy_fun = model.original_energy
    elif energy == "E3_y":
        energy_fun = model.energy_E3_y
    else:
        raise ValueError("energy должен быть 'original' или 'E3_y'.")

    rows = []
    E0 = energy_fun(y0)
    for c in c_values:
        c_vec = np.array([c], dtype=float)
        y = y0 + V @ c_vec
        E = energy_fun(y)
        rows.append(
            {
                "c": float(c),
                "E": float(E),
                "E_minus_E0": float(E - E0),
                **{f"y{i+1}": float(v) for i, v in enumerate(y)},
            }
        )
    return pd.DataFrame(rows)


def bias_sensitivity_scan(
    model: HopfieldModel,
    y0: Array,
    bias_direction: Array,
    deltas: Iterable[float],
    energy: str = "original",
) -> pd.DataFrame:
    """
    Проверка чувствительности энергии к сдвигу b в мёртвом направлении.

    Для ReLU, если g_i(y0)=0, исходная энергия не зависит от b_i, потому что
    член -(b_i)g_i(y0) равен нулю. Это численно демонстрирует замечание о том,
    что энергия в плоских регионах теряет часть информации о состоянии/сдвиге.
    """
    y0 = np.asarray(y0, dtype=float)
    direction = np.asarray(bias_direction, dtype=float)
    direction = direction / max(norm(direction), 1e-12)

    rows = []
    for delta in deltas:
        shifted = HopfieldModel(model.W, model.b + float(delta) * direction, model.activation)
        if energy == "original":
            E = shifted.original_energy(y0)
        elif energy == "E3_y":
            E = shifted.energy_E3_y(y0)
        else:
            raise ValueError("energy должен быть 'original' или 'E3_y'.")
        rows.append({"delta_b": float(delta), "E": float(E)})
    return pd.DataFrame(rows)


def spectral_diagnostics(
        model: HopfieldModel,
        y: Array,
        tol: float = 1e-9)\
        -> Dict[str, object]:
    """
    Спектральные матрицы из замечаний к численному моделированию.

    Возвращаются спектры:
    1. W_perp Lambda_perp - I в активном базисе Range Lambda;
    2. Lambda - Lambda W Lambda;
    3. Lambda W Lambda - Lambda.

    Пункт 1 связан с линейной устойчивостью активной подсистемы, пункты 2--3
    показывают отличие энергетического критерия от динамической устойчивости.
    """

    y = np.asarray(y, dtype=float)
    Lam = model.Lambda(y)
    info = orthogonal_projectors_from_lambda(Lam, tol=tol)
    Qa = info["Qa"]
    if Qa.size:
        Wp = restrict_to_basis(model.W, Qa)
        Lamp = restrict_to_basis(Lam, Qa)
        A_active = Wp @ Lamp - np.eye(Qa.shape[1])
        eig_active = np.linalg.eigvals(A_active)
    else:
        A_active = np.zeros((0, 0))
        eig_active = np.array([])

    H_energy = Lam - Lam @ model.W @ Lam
    H_energy_negative = -H_energy
    return {
        "y": y,
        "rank_Lambda": info["rank"],
        "nullity_Lambda": info["nullity"],
        "Lambda_eigenvalues": info["eigenvalues"],
        "eig_Wperp_Lambdaperp_minus_I": eig_active,
        "eig_Lambda_minus_Lambda_W_Lambda": np.linalg.eigvalsh(symmetrize(H_energy)),
        "eig_Lambda_W_Lambda_minus_Lambda": np.linalg.eigvalsh(symmetrize(H_energy_negative)),
        "Pa": info["Pa"],
        "Pd": info["Pd"],
    }


# -----------------------------------------------------------------------------
# 6. Интегрирование ОДУ, кластеризация финальных состояний, метрики сходимости
# -----------------------------------------------------------------------------

def integrate_trajectory(
    field: Callable[[float, Array], Array],
    y0: Array,
    t_span: Tuple[float, float] = (0.0, 25.0),
    t_eval: Optional[Array] = None,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> solve_ivp:
    """
    Интегрировать ОДУ с безопасными настройками точности.
    """
    y0 = np.asarray(y0, dtype=float)
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 400)
    return solve_ivp(field, t_span, y0, t_eval=t_eval, rtol=rtol, atol=atol)

def cluster_points(points: Array, radius: float = 1e-2) -> Tuple[Array, Array]:
    """
    Простейшая кластеризация конечных точек.

    Возвращает центры кластеров и метку кластера для каждой точки. Алгоритм
    жадный, но достаточен для первичного отчёта об областях притяжения.
    """

    points = np.asarray(points, dtype=float)
    centers: List[Array] = []
    labels = np.full(points.shape[0], -1, dtype=int)
    for i, p in enumerate(points):
        assigned = False
        for k, c in enumerate(centers):
            if norm(p - c) <= radius:
                labels[i] = k
                # Обновляем центр как среднее уже найденных точек кластера.
                members = points[labels == k]
                centers[k] = np.mean(np.vstack([members, p]), axis=0)
                assigned = True
                break
        if not assigned:
            labels[i] = len(centers)
            centers.append(p.copy())
    if not centers:
        return np.zeros((0, points.shape[1])), labels
    return np.vstack(centers), labels


def integrate_fixed_rk4(
    field: Callable[[float, Array], Array],
    y0: Array,
    t_final: float = 15.0,
    dt: float = 0.03,
    stop_callback: Optional[Callable[[float, Array], bool]] = None,
    blowup_limit: float = 1e6,
    save_trajectory: bool = False,
) -> Dict[str, object]:
    """
    Безопасное интегрирование фиксированным шагом методом Рунге--Кутты 4-го порядка.

    Эта функция не зависает, потому что всегда делает конечное число шагов.
    """
    y = np.asarray(y0, dtype=float).copy()
    t = 0.0

    if dt <= 0:
        raise ValueError("dt должен быть положительным.")

    n_steps = int(np.ceil(t_final / dt))

    if save_trajectory:
        t_values = [t]
        y_values = [y.copy()]
    else:
        t_values = None
        y_values = None

    status = "finished"

    for _ in range(n_steps):
        h = min(dt, t_final - t)

        if h <= 0:
            break

        try:
            k1 = np.asarray(field(t, y), dtype=float)
            k2 = np.asarray(field(t + 0.5 * h, y + 0.5 * h * k1), dtype=float)
            k3 = np.asarray(field(t + 0.5 * h, y + 0.5 * h * k2), dtype=float)
            k4 = np.asarray(field(t + h, y + h * k3), dtype=float)

            y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        except Exception as exc:
            status = f"failed: {type(exc).__name__}: {exc}"
            break

        if (not np.all(np.isfinite(y_next))) or norm(y_next) > blowup_limit:
            y = y_next
            t = t + h
            status = "blowup"
            break

        y = y_next
        t = t + h

        if save_trajectory:
            t_values.append(t)
            y_values.append(y.copy())

        if stop_callback is not None and stop_callback(t, y):
            status = "event_converged"
            break

    result: Dict[str, object] = {
        "t_final_reached": float(t),
        "y_final": y,
        "status": status,
        "success": status in {"finished", "event_converged"},
    }

    if save_trajectory:
        result["t_values"] = np.asarray(t_values, dtype=float)
        result["y_values"] = np.asarray(y_values, dtype=float)

    return result

def _safe_energy_value(energy_fun: Callable[[Array], float], y: Array) -> float:
    try:
        value = float(energy_fun(y))
        if np.isfinite(value):
            return value
        return np.nan
    except Exception:
        return np.nan

def compare_original_modified(
    model: HopfieldModel,
    initial_conditions: Array,
    t_final: float = 15.0,
    residual_tol: float = 1e-4,
    equilibrium_tol: float = 5e-3,
    dt: float = 0.03,
    blowup_limit: float = 1e6,
    equilibria_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Сравнивает исходную и модифицированную динамику.

    Теперь успех считается двумя способами:
    1. по малой исходной невязке:
        ||Wg(y_T)-y_T+b|| < residual_tol;
    2. по близости к найденному исходному равновесию:
        dist(y_T, Equilibria) < equilibrium_tol.

    Это устойчивее, чем только проверка невязки в конечный момент времени.
    """
    rows = []

    initial_conditions = np.asarray(initial_conditions, dtype=float)

    if equilibria_df is None:
        equilibria_df = find_equilibria_2d(model, dynamics="original")

    experiments = [
        ("original", model.original_field, model.original_energy),
        ("modified", model.modified_y_field, model.energy_E3_y),
    ]

    for idx, y0 in enumerate(initial_conditions):
        for name, field, energy_fun in experiments:

            def stop_callback(_t: float, y: Array) -> bool:
                residual_ok = norm(model.original_residual(y)) < residual_tol
                eq_dist_ok = _nearest_equilibrium_distance(y, equilibria_df) < equilibrium_tol
                return bool(residual_ok or eq_dist_ok)

            result = integrate_fixed_rk4(
                field=field,
                y0=y0,
                t_final=t_final,
                dt=dt,
                stop_callback=stop_callback,
                blowup_limit=blowup_limit,
                save_trajectory=False,
            )

            yT = np.asarray(result["y_final"], dtype=float)

            residual_norm = float(norm(model.original_residual(yT)))
            nearest_eq_distance = _nearest_equilibrium_distance(yT, equilibria_df)

            residual_success = residual_norm < residual_tol
            distance_success = nearest_eq_distance < equilibrium_tol

            energy_initial = _safe_energy_value(energy_fun, y0)
            energy_final = _safe_energy_value(energy_fun, yT)

            rows.append(
                {
                    "ic_id": int(idx),
                    "dynamics": name,
                    "integrator": "fixed_rk4",
                    "status": str(result["status"]),
                    "t_final_reached": float(result["t_final_reached"]),
                    "success": bool(residual_success or distance_success),
                    "residual_success": bool(residual_success),
                    "distance_success": bool(distance_success),
                    "residual_norm": residual_norm,
                    "nearest_eq_distance": float(nearest_eq_distance),
                    "energy_initial": energy_initial,
                    "energy_final": energy_final,
                    "energy_drop": (
                        energy_initial - energy_final
                        if np.isfinite(energy_initial) and np.isfinite(energy_final)
                        else np.nan
                    ),
                    **{f"y0_{i+1}": float(v) for i, v in enumerate(y0)},
                    **{f"yT_{i+1}": float(v) for i, v in enumerate(yT)},
                }
            )

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 7. Малые 2D модели для репликации графиков ReLU/softmax
# -----------------------------------------------------------------------------

def make_demo_model(activation_name: str) -> HopfieldModel:
    """
    Симметричная двумерная модель для рисунков и тестов.

    Параметры выбраны не как обученная память, а как устойчивый демонстрационный
    пример: исходная энергия показывает плоскости/полосы, а E3 даёт изолинии без
    неконтролируемых плоских направлений.
    """

    act = get_activation(activation_name)
    if activation_name.lower() == "softmax":
        W = np.array([[1.35, -0.35], [-0.35, 1.35]], dtype=float)
        b = np.array([0.0, 0.0], dtype=float)
    else:
        W = np.array([[0.85, 0.25], [0.25, 0.85]], dtype=float)
        b = np.array([0.10, -0.05], dtype=float)
    return HopfieldModel(W=W, b=b, activation=act)

def make_random_demo_model(
    activation_name: str,
    n: int = 2,
    w_norm: float = 0.65,
    b_scale: float = 0.35,
    symmetric: bool = True,
    seed: Optional[int] = None,
) -> HopfieldModel:
    """
    Создаёт демо-модель со случайной матрицей W.

    Если seed=None, то при каждом новом запуске W будет новой.
    Если seed задан, эксперимент будет воспроизводимым.

    Parameters
    ----------
    activation_name:
        "relu" или "softmax".
    n:
        Размерность фазового пространства.
    w_norm:
        Желаемая спектральная норма W.
    b_scale:
        Масштаб случайного смещения b.
    symmetric:
        Если True, W делается симметричной.
    seed:
        Seed генератора. None означает новую случайную W при каждом запуске.
    """
    rng = np.random.default_rng(seed)

    A = rng.normal(size=(n, n))

    if symmetric:
        W = 0.5 * (A + A.T)
    else:
        W = A

    W = _normalize_spectral_norm(W, target_norm=w_norm)

    b = rng.normal(scale=b_scale, size=n)

    name = activation_name.lower()

    if name == "relu":
        activation = relu_activation()
    elif name == "softmax":
        activation = softmax_activation()
    else:
        raise ValueError("activation_name должен быть 'relu' или 'softmax'.")

    return HopfieldModel(W=W, b=b, activation=activation)

def sample_initial_conditions(
    n_samples: int,
    n_dim: int,
    low: float = -3.0,
    high: float = 3.0,
    seed: int = 42,
) -> Array:
    """
    Воспроизводимая выборка начальных условий.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n_samples, n_dim))

def _cluster_points(points: list[Array], tol: float = 1e-5) -> list[Array]:
    """
    Убирает дубликаты среди найденных корней.
    """
    clusters: list[Array] = []

    for p in points:
        p = np.asarray(p, dtype=float)

        if not np.all(np.isfinite(p)):
            continue

        is_new = True

        for q in clusters:
            if norm(p - q) < tol:
                is_new = False
                break

        if is_new:
            clusters.append(p)

    return clusters


def find_equilibria_2d(
    model: HopfieldModel,
    dynamics: str = "original",
    xlim: tuple[float, float] = (-4.0, 4.0),
    ylim: tuple[float, float] = (-4.0, 4.0),
    grid_size: int = 13,
    root_tol: float = 1e-9,
    cluster_tol: float = 1e-5,
) -> pd.DataFrame:
    """
    Ищет равновесия двумерной системы из разных начальных приближений.

    Для исходной системы равновесия решают:
        W g(y) - y + b = 0.

    Для модифицированной системы равновесия решают:
        R(y)(W g(y) - y + b) = 0.

    Важно:
    если R(y) вырождена, модифицированная система может иметь ложные
    равновесия, для которых исходная невязка не равна нулю.
    Поэтому в таблице сохраняются обе нормы:
        original_residual_norm и modified_field_norm.
    """
    if model.n != 2:
        raise ValueError("find_equilibria_2d реализована только для n=2.")

    dynamics_key = dynamics.lower()

    if dynamics_key == "original":
        vector_field = lambda y: model.original_residual(y)
    elif dynamics_key in {"modified", "modified_y"}:
        vector_field = lambda y: model.modified_y_field(0.0, y)
    else:
        raise ValueError("dynamics должен быть 'original' или 'modified'.")

    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)

    roots: list[Array] = []

    for x in xs:
        for y in ys:
            start = np.array([x, y], dtype=float)

            try:
                sol = root(vector_field, start, method="hybr", tol=root_tol)
            except Exception:
                continue

            if not sol.success:
                continue

            candidate = np.asarray(sol.x, dtype=float)

            if not np.all(np.isfinite(candidate)):
                continue

            field_norm = norm(vector_field(candidate))

            if field_norm < 1e-6:
                roots.append(candidate)

    roots = _cluster_points(roots, tol=cluster_tol)

    rows = []

    for idx, y_star in enumerate(roots):
        original_residual = model.original_residual(y_star)
        modified_field = model.modified_y_field(0.0, y_star)

        original_residual_norm = float(norm(original_residual))
        modified_field_norm = float(norm(modified_field))

        rows.append(
            {
                "eq_id": int(idx),
                "dynamics_used_for_search": dynamics_key,
                "y1": float(y_star[0]),
                "y2": float(y_star[1]),
                "original_residual_norm": original_residual_norm,
                "modified_field_norm": modified_field_norm,
                "is_original_equilibrium": bool(original_residual_norm < 1e-6),
                "is_modified_equilibrium": bool(modified_field_norm < 1e-6),
            }
        )

    return pd.DataFrame(rows)

def _nearest_equilibrium_distance(y: Array, equilibria_df: pd.DataFrame) -> float:
    """
    Расстояние от точки y до ближайшего исходного равновесия.
    """
    if equilibria_df.empty:
        return np.inf

    original_eq = equilibria_df[equilibria_df["is_original_equilibrium"]]

    if original_eq.empty:
        return np.inf

    points = original_eq[["y1", "y2"]].to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)

    return float(np.min(np.linalg.norm(points - y[None, :], axis=1)))

# -----------------------------------------------------------------------------
# 8. Утилиты сохранения результатов
# -----------------------------------------------------------------------------

def save_json_like_diagnostics(diags: Dict[str, object], path: str | Path) -> None:
    """
    Сохранить диагностики в текстовом формате, совместимом с отчётом.

    JSON со сложными числами и ndarray неудобен, поэтому пишем читаемый .txt.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for key, value in diags.items():
            f.write(f"{key}:\n")
            if isinstance(value, np.ndarray):
                f.write(np.array2string(value, precision=6, suppress_small=True))
            else:
                f.write(str(value))
            f.write("\n\n")
