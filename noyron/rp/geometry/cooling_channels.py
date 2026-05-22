"""
cooling_channels.py
===================

SDF voids for axial rectangular cooling channels carved into the bilayer
chamber wall.

Geometry model
--------------
Each channel is a rectangular cross-section slot, running axially from
z_min to z_max, with:

  - Radial extent : from r_liner_OD(z) to r_liner_OD(z) + h_channel(z)
  - Circumferential extent : arc angle dtheta_ch(z) = width(z) / r_mean(z)
    where r_mean = r_liner_OD + h_channel/2

Channels are equally spaced by 2π/n_channels around the z-axis.

The modular angular formulation collapses all n_channels SDFs into one
evaluation by mapping any (r, θ, z) point to the nearest channel sector:

    φ  = 2π / n_channels          (sector half-width × 2)
    θ_mod = ((θ mod φ) + φ) mod φ  (map θ into [0, φ))
    θ_loc = θ_mod - φ/2            (centre sector on zero)

Then |θ_loc| ≤ dtheta_ch(z)/2 is inside the channel circumferentially.

SDF convention: f < 0 inside the channel void, f > 0 outside.
The final chamber-with-channels SDF uses difference():
    f_result = max(f_chamber, -f_channels)

Sources
-------
Quilez, I. (2023). "Signed Distance Functions."
    iquilezles.org/articles/distfunctions.
    — box SDF formula; smooth operations.

NASA SP-8087 (1977). "Liquid Rocket Engine Fluid-Cooled Combustion
    Chambers." Sec. 3.1 — rectangular channel cross-section geometry.

Huzel & Huang (1992). "Modern Engineering for Design of Liquid-Propellant
    Rocket Engines." NASA SP-125. p. 76 — channel layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

from noyron.kernel.geometry.sdf.operations import difference
from noyron.rp.geometry.chamber_nozzle import chamber_nozzle_sdf

if TYPE_CHECKING:
    from noyron.rp.nozzle.rao import RaoNozzle
    from noyron.rp.regen.coolant import CoolingChannelGeometry
    from noyron.rp.structural.bilayer import BilayerWall

# Number of axial sample points for pre-sampling non-vectorisable lambdas
_N_SAMPLES = 500


def _build_interp(func: Callable[[float], float], z_arr: np.ndarray):
    """Pre-sample a scalar callable and return a vectorised interp1d."""
    vals = np.array([func(float(z)) for z in z_arr], dtype=np.float64)
    return interp1d(
        z_arr, vals,
        kind="linear",
        bounds_error=False,
        fill_value=(vals[0], vals[-1]),
    )


def axial_channel_sdf(
    *,
    channel_geometry: "CoolingChannelGeometry",
    liner_OD: Callable[[float], float],
    channel_index: int,
    n_channels: int,
    z_min: float,
    z_max: float,
) -> Callable:
    """
    SDF of a single axial rectangular cooling channel void.

    The channel is indexed by ``channel_index`` ∈ {0, …, n_channels-1},
    centred at angular position θ_0 = 2π × channel_index / n_channels.

    Parameters
    ----------
    channel_geometry : CoolingChannelGeometry
        Channel width, height, wall thickness, and land thickness functions.
    liner_OD : Callable[[float], float]
        Outer diameter of the liner (= inner radius of channels) [m] as a
        function of axial coordinate z.
    channel_index : int
        Which channel (0-based).
    n_channels : int
        Total number of channels.
    z_min, z_max : float
        Axial extent of the channel domain [m].

    Returns
    -------
    callable  f(x, y, z) -> ndarray
        SDF; negative inside the channel, positive outside.
    """
    theta_0 = 2.0 * np.pi * channel_index / n_channels
    z_arr = np.linspace(z_min, z_max, _N_SAMPLES)

    # Pre-sample non-vectorisable callables
    _r_inner = _build_interp(liner_OD, z_arr)           # liner OD = channel floor
    _h_ch = _build_interp(channel_geometry.height, z_arr)
    _w_ch = _build_interp(channel_geometry.width, z_arr)

    def _f(x, y, z):
        x_ = np.asarray(x, dtype=np.float32)
        y_ = np.asarray(y, dtype=np.float32)
        z_ = np.asarray(z, dtype=np.float32)

        rho = np.sqrt(x_ * x_ + y_ * y_)
        theta = np.arctan2(y_.astype(np.float64), x_.astype(np.float64))

        # Angular distance from channel centre
        dtheta = theta - theta_0
        # Wrap to (-π, π]
        dtheta = (dtheta + np.pi) % (2.0 * np.pi) - np.pi

        z_c = np.clip(z_.astype(np.float64), z_min, z_max)
        r_in = _r_inner(z_c).astype(np.float32)
        h = _h_ch(z_c).astype(np.float32)
        w = _w_ch(z_c).astype(np.float32)
        r_mean = r_in + h * 0.5
        # Half-angle subtended by channel width at r_mean
        dtheta_half = (w / (2.0 * r_mean)).astype(np.float64)

        # Radial signed distance (negative when r_in ≤ rho ≤ r_in+h)
        rho32 = rho.astype(np.float64)
        f_rad = np.maximum(r_in.astype(np.float64) - rho32,
                           rho32 - (r_in + h).astype(np.float64))

        # Angular signed distance (negative when |dtheta| ≤ dtheta_half)
        f_ang = (np.abs(dtheta) - dtheta_half) * r_mean.astype(np.float64)

        # Axial signed distance (negative when z_min ≤ z ≤ z_max)
        f_axial = np.maximum(z_min - z_.astype(np.float64),
                             z_.astype(np.float64) - z_max)

        return np.maximum(np.maximum(f_rad, f_ang), f_axial).astype(np.float32)

    return _f


def all_channels_sdf(
    *,
    channel_geometry: "CoolingChannelGeometry",
    liner_OD: Callable[[float], float],
    n_channels: int,
    z_min: float,
    z_max: float,
) -> Callable:
    """
    SDF of all n_channels axial cooling channel voids combined.

    Uses the modular angular formulation so cost is O(1) in n_channels:

        φ        = 2π / n_channels
        θ_mod    = ((θ mod φ) + φ) mod φ    ∈ [0, φ)
        θ_local  = θ_mod - φ/2              ∈ [-φ/2, +φ/2)

    A point is inside ANY channel iff |θ_local| ≤ dtheta_ch(z)/2  and
    r_in(z) ≤ ρ ≤ r_in(z)+h(z) and z_min ≤ z ≤ z_max.

    Parameters
    ----------
    channel_geometry : CoolingChannelGeometry
    liner_OD : Callable[[float], float]
        Outer diameter of the liner (channel floor radius) [m].
    n_channels : int
    z_min, z_max : float
        Axial extent [m].

    Returns
    -------
    callable  f(x, y, z) -> ndarray
        SDF; negative inside any channel, positive outside all channels.
    """
    phi = 2.0 * np.pi / n_channels   # sector width

    z_arr = np.linspace(z_min, z_max, _N_SAMPLES)
    _r_inner = _build_interp(liner_OD, z_arr)
    _h_ch = _build_interp(channel_geometry.height, z_arr)
    _w_ch = _build_interp(channel_geometry.width, z_arr)

    phi32 = np.float32(phi)
    z_min32 = np.float32(z_min)
    z_max32 = np.float32(z_max)

    def _f(x, y, z):
        # Keep all intermediates in float32 to cap memory at ~4 bytes/voxel.
        # Scipy interpolants return float64; cast immediately after each call.
        x_ = np.asarray(x, dtype=np.float32)
        y_ = np.asarray(y, dtype=np.float32)
        z_ = np.asarray(z, dtype=np.float32)

        rho = np.sqrt(x_ * x_ + y_ * y_)
        theta = np.arctan2(y_, x_)

        # Map θ into nearest sector centre via modular arithmetic.
        # theta_mod ∈ [0, phi).  If theta_mod < phi/2 the nearest channel
        # is at 0 and the local offset is +theta_mod.  If theta_mod ≥ phi/2
        # the nearest channel is at phi and the local offset is
        # theta_mod - phi (negative).  This makes |theta_local| the actual
        # angular distance to the nearest channel centre.
        theta_mod = np.mod(theta, phi32)                              # [0, phi)
        theta_local = np.where(theta_mod < phi32 * np.float32(0.5),  # (-phi/2, phi/2]
                               theta_mod,
                               theta_mod - phi32)

        z_c = np.clip(z_, z_min32, z_max32)
        r_in = _r_inner(z_c).astype(np.float32)
        h = _h_ch(z_c).astype(np.float32)
        w = _w_ch(z_c).astype(np.float32)
        r_mean = r_in + h * np.float32(0.5)
        dtheta_half = w / (np.float32(2.0) * r_mean)

        f_rad = np.maximum(r_in - rho, rho - (r_in + h))
        f_ang = (np.abs(theta_local) - dtheta_half) * r_mean
        f_axial = np.maximum(z_min32 - z_, z_ - z_max32)

        return np.maximum(np.maximum(f_rad, f_ang), f_axial)

    return _f


def chamber_with_channels_sdf(
    *,
    rao_nozzle: "RaoNozzle",
    bilayer: "BilayerWall",
    L_chamber: float | None = None,
    n_profile_points: int = 500,
) -> tuple[Callable, dict]:
    """
    SDF of the outer jacket with cooling channel voids carved in.

    Returns the difference: outer jacket solid minus all channel voids.

        f_result(p) = max(f_jacket(p), -f_channels(p))

    Channel domain: spans from z_chamber_start to z_exit (full axial extent
    of the chamber + nozzle).

    Parameters
    ----------
    rao_nozzle : RaoNozzle
    bilayer : BilayerWall
    L_chamber : float or None
        Override for cylindrical chamber length upstream of the convergent [m].
        If None (default), ``rao_nozzle.L_chamber`` is used.
    n_profile_points : int
        Axial sample count for profile interpolants.

    Returns
    -------
    sdf : callable  f(x, y, z) -> ndarray
        SDF of the channelled wall; negative inside the solid walls,
        positive outside.
    profile : dict
        Same keys as ``chamber_nozzle_sdf``, plus:
          'n_channels' : int
          'z_ch_start' : float  [m]  axial start of channel domain
    """
    # Build outer jacket SDF and profile
    f_jacket, profile = chamber_nozzle_sdf(
        rao_nozzle=rao_nozzle,
        bilayer=bilayer,
        L_chamber=L_chamber,
        n_profile_points=n_profile_points,
    )

    z_min = profile["z_min"]
    z_max = profile["z_max"]
    z_ax = profile["z_ax"]
    r_inner = profile["r_inner"]

    # Vectorised liner_OD from the precomputed profile
    _liner_OD_interp = interp1d(
        z_ax, r_inner,
        kind="linear",
        bounds_error=False,
        fill_value=(float(r_inner[0]), float(r_inner[-1])),
    )

    def liner_OD(z: float) -> float:
        return float(_liner_OD_interp(z))

    geom = bilayer.channel_geometry
    n_ch = geom.n_channels

    f_channels = all_channels_sdf(
        channel_geometry=geom,
        liner_OD=liner_OD,
        n_channels=n_ch,
        z_min=z_min,
        z_max=z_max,
    )

    f_result = difference(f_jacket, f_channels)

    profile["n_channels"] = n_ch
    profile["z_ch_start"] = z_min

    return f_result, profile
