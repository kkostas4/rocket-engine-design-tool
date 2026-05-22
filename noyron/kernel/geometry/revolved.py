"""
revolved.py
===========

Domain-agnostic SDF primitives for revolved / extruded bodies.

Used for inlet/outlet manifold rings and boss stubs in both the
rocket-propulsion (noyron.rp) and heat-exchanger (noyron.hx) layers.

SDF convention: f < 0 inside solid, f > 0 outside, f = 0 on surface.

Source: Quilez, I. (2023). "Signed Distance Functions."
        iquilezles.org/articles/distfunctions — cylinder and revolution.
"""

from __future__ import annotations

import numpy as np


def annular_shell_sdf(
    *,
    r_inner: float,
    r_outer: float,
    z_lo: float,
    z_hi: float,
    wall_thickness: float,
):
    """
    Hollow annular shell: solid ring with internal fluid cavity.

    The outer shape is a capped annular ring (washer).  The interior is
    hollowed out by wall_thickness on every side — radially and axially —
    leaving a fluid-accessible cavity.

    Solid region:
        { r_inner ≤ ρ ≤ r_outer,  z_lo ≤ z ≤ z_hi }
    Cavity (fluid void):
        { r_inner+t ≤ ρ ≤ r_outer−t,  z_lo+t ≤ z ≤ z_hi−t }
    where t = wall_thickness.

    If the wall is so thick that no cavity is geometrically possible
    (r_outer − r_inner < 2t or z_hi − z_lo < 2t) the function returns
    a fully solid ring with no void.

    Parameters
    ----------
    r_inner : float
        Inner bore radius [m].
    r_outer : float
        Outer radius of the ring [m].
    z_lo : float
        Axial lower bound [m].
    z_hi : float
        Axial upper bound [m].
    wall_thickness : float
        Uniform shell wall thickness applied to all faces [m].

    Returns
    -------
    Callable f(x, y, z) -> ndarray
        SDF — negative inside solid shell material.

    Source: Quilez, I. (2023). "Signed Distance Functions."
            iquilezles.org/articles/distfunctions — annular ring SDF.
    Validity: exact on radial cross-sections; approximate near axial caps
              (same caveat as revolution.py revolve()).
    """
    if r_outer <= r_inner:
        raise ValueError(f"r_outer ({r_outer:.6f}) must exceed r_inner ({r_inner:.6f})")
    if z_hi <= z_lo:
        raise ValueError(f"z_hi ({z_hi:.6f}) must exceed z_lo ({z_lo:.6f})")

    r_cav_inner = r_inner + wall_thickness
    r_cav_outer = r_outer - wall_thickness
    z_cav_lo = z_lo + wall_thickness
    z_cav_hi = z_hi - wall_thickness
    _has_cavity = (r_cav_outer > r_cav_inner) and (z_cav_hi > z_cav_lo)

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        z_ = np.asarray(z, dtype=float)
        rho = np.sqrt(x_ ** 2 + y_ ** 2)

        # Solid washer: r_inner ≤ ρ ≤ r_outer, z_lo ≤ z ≤ z_hi
        f_shell = np.maximum(
            np.maximum(rho - r_outer, r_inner - rho),
            np.maximum(z_lo - z_, z_ - z_hi),
        )

        if not _has_cavity:
            return f_shell

        # Internal fluid cavity
        f_cav = np.maximum(
            np.maximum(rho - r_cav_outer, r_cav_inner - rho),
            np.maximum(z_cav_lo - z_, z_ - z_cav_hi),
        )
        # Shell solid = washer minus cavity (difference SDF)
        return np.maximum(f_shell, -f_cav)

    return _f


def cylindrical_boss_sdf(
    *,
    axis_origin: tuple[float, float, float],
    axis_direction: tuple[float, float, float],
    OD: float,
    length: float,
    inner_OD: float | None = None,
):
    """
    Capped cylinder (boss / port stub) with an arbitrary axis direction.

    The cylinder starts at axis_origin and extends length metres along
    axis_direction.  If inner_OD is given the boss is a hollow tube.

    Parameters
    ----------
    axis_origin : (ox, oy, oz)
        Start point of the cylinder axis [m].
    axis_direction : (dx, dy, dz)
        Direction of the cylinder axis.  Normalised internally.
    OD : float
        Outer diameter of the boss [m].
    length : float
        Axial length of the boss [m].
    inner_OD : float or None
        If provided, creates a hollow tube with this inner diameter [m].
        Must be < OD.  If ≤ 0, ignored (treated as None).

    Returns
    -------
    Callable f(x, y, z) -> ndarray
        SDF — negative inside the boss solid material.

    Source: Quilez, I. (2023). "Signed Distance Functions."
            iquilezles.org/articles/distfunctions — capped cylinder SDF.
    """
    O = np.asarray(axis_origin, dtype=float)
    d = np.asarray(axis_direction, dtype=float)
    d = d / np.linalg.norm(d)
    R = OD / 2.0
    R_inner = (inner_OD / 2.0) if (inner_OD is not None and inner_OD > 1e-9) else None

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=float)
        y_ = np.asarray(y, dtype=float)
        z_ = np.asarray(z, dtype=float)

        vx = x_ - O[0]
        vy = y_ - O[1]
        vz = z_ - O[2]

        # Axial projection along boss direction
        t = vx * d[0] + vy * d[1] + vz * d[2]

        # Perpendicular (radial) distance from boss axis
        px = vx - t * d[0]
        py = vy - t * d[1]
        pz = vz - t * d[2]
        rho = np.sqrt(px ** 2 + py ** 2 + pz ** 2)

        # Capped outer cylinder (negative inside)
        f = np.maximum(np.maximum(rho - R, -t), t - length)

        if R_inner is not None:
            # Hollow tube: subtract inner bore
            f_inner = np.maximum(np.maximum(rho - R_inner, -t), t - length)
            f = np.maximum(f, -f_inner)

        return f

    return _f
