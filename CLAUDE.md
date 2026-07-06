# Noyron-Clone — Methalox Computational Engine Model

## Project Purpose

Reverse-engineer a Noyron-style (LEAP 71) computational engine model for a small
methane/LOX rocket engine. Each subsystem is independently modelled, verified
against published correlations, and integrated through a geometry kernel.

This is an engineering tool, not a toy. Every number must be traceable.

---

## Reference Engine Design — Authoritative Numbers

These numbers are the **frozen baseline** for the project. All subsystems must
size against this engine unless explicitly varying a parameter for trade study.

### Top-Level
| Parameter | SI | US |
|---|---|---|
| Thrust | 4,448.22 N | 1,000 lbf |
| Chamber pressure (Pc) | 6.8948 MPa (68.95 bar) | 1,000 psia |
| Propellants | LCH4 / LOX | — |
| O/F (mass) | 3.60 | 3.60 |
| Expansion ratio (Ae/At) | 10.605 | 10.605 |
| Isp (delivered, SL-optimised) | 318.44 s | 318.44 s |
| c* | 1,893.3 m/s | 5,979 ft/s |
| c (effective exhaust velocity) | 3,123.9 m/s | 12,160.8 ft/s |

### Mass Flow
| Stream | kg/s |
|---|---|
| Total | 1.4239 |
| LOX | 1.1144 |
| LCH4 | 0.3096 |

### Geometry
| Feature | Value (m) | Value (mm) |
|---|---|---|
| Throat area | 3.766e-4 m² | — |
| Throat diameter | 0.021895 | 21.895 |
| Exit area | 4.147e-3 m² | — |
| Exit diameter | 0.072662 | 72.662 |
| Ac/At | 4.0 | — |
| Chamber area | 1.564e-3 m² | — |
| Chamber diameter | 0.044625 | 44.625 |
| L* (characteristic length) | 0.500 | 500 |
| Chamber length | 0.125 | 125 |

> **Note on throat geometry:** The throat diameter above (21.895 mm, r_throat = 10.947 mm)
> is the CEA-corrected value from equilibrium expansion at the methalox_5kN operating
> point.  Earlier versions of this document listed 22.313 mm (r = 11.156 mm) from a
> pre-CEA spreadsheet calculation; that value is stale.  All test fixtures and the
> engine model use 10.947 mm.

### Injector — Fuel (LCH4)
| Parameter | SI | US |
|---|---|---|
| Injection pressure | 9.997 MPa | 1,450 psia |
| ΔP across injector | 3.103 MPa | 450 psi |
| ΔP/Pc | 0.31 (31%) | — |
| Injection temperature | 118 K | -240 °F |
| Injection density | 386 kg/m³ | 24.1 lbm/ft³ |
| Discharge coefficient (Cd) | 0.6 | — |
| Total orifice area | 1.054e-5 m² | — |
| Number of orifices | 20 | — |
| Individual orifice diameter | 0.000819 m | 0.819 mm |

### Injector — Oxidiser (LOX)
| Parameter | SI | US |
|---|---|---|
| Injection pressure | 9.997 MPa | 1,450 psia |
| ΔP across injector | 3.103 MPa | 450 psi |
| Injection temperature | 100 K | -280 °F |
| Injection density | 1,116 kg/m³ | — |
| Discharge coefficient (Cd) | 0.6 | — |
| Total annulus area | 2.232e-5 m² | — |
| Pintle outer diameter | 0.030 m | 30 mm |
| Annulus outer diameter | 0.030470 m | 30.47 mm |
| Injector type | Pintle (single element) | — |

### CEA Sweep (Pc=1000 psia, ε=68.027) — Reference Table
The full sweep is in `physics/thermochem/reference_data/cea_sweep_pc1000.csv`.
Peak Isp = 377.97 s at O/F = 3.60. The delivered Isp at ε=10.605 is 318.4 s.

---

## Project Rules — Non-Negotiable

### Units
- **Internal**: Strictly SI (Pa, m, kg, s, K, N). No mixed units inside any function.
- **Boundary I/O**: Engineering units (psia, mm, lbf, °F) allowed only at
  user-facing reports and when reading legacy data. Convert at the boundary,
  never in the middle of a calculation.
- **Constants module**: All unit conversion factors live in
  `physics/util/units.py`. No magic numbers in physics code.

### Correlations and Coefficients
- **Never fabricate a correlation, coefficient, or material property.**
- Every physics function must cite its source in the docstring. Format:
  ```python
  """
  Bartz gas-side heat transfer coefficient.

  Source: Bartz, D.R. (1957). "A Simple Equation for Rapid Estimation
          of Rocket Nozzle Convective Heat Transfer Coefficients."
          Jet Propulsion, Vol. 27, No. 1, pp. 49-51.
  Equation: Eq. (5)
  Validity: subsonic chamber + supersonic nozzle, gamma 1.1-1.3, Pr ~0.5-1.0
  """
  ```
- If a value is unsourced, **flag it loudly** with `# TODO(unsourced)` and a
  comment explaining where you got it from. Do not silently insert a number.

### Verification
- Every physics function has a sibling test in `verification/`.
- Every recalculation of a baseline number has a Python script in `calcs/`
  that reproduces the spreadsheet value to ≤ 0.5% error.
- Every new correlation must include a regression test that locks in its output
  for a known input.

### Code Style
- Python 3.11+. Type hints everywhere.
- Numpy for arrays, scipy for solvers. No pandas in physics code.
- Functions over classes unless state genuinely matters (a chamber geometry,
  a coolant channel network).
- Docstrings: numpy style. First line is a one-sentence summary.

### Commits
- One subsystem per commit when possible.
- Commit message format: `<subsystem>: <verb> <what>`
  Example: `regen: add Bartz correlation with verification test`

---

## Commands

### Run all verification tests
```bash
pytest verification/ -v
```

### Run subsystem tests
```bash
pytest verification/test_regen.py -v
pytest verification/test_nozzle.py -v
```

### Verify baseline engine numbers
```bash
python calcs/verify_baseline.py
```

### Format and lint
```bash
ruff format .
ruff check .
mypy physics/
```

---

## Subsystem Boundaries

Each subsystem owns its folder. Cross-subsystem communication happens through
**explicit interfaces** (function signatures, dataclasses), never through
shared mutable state.

| Subsystem | Folder | Owns |
|---|---|---|
| Thermochemistry | `physics/thermochem/` | CEA wrapper, propellant properties |
| Nozzle | `physics/nozzle/` | MOC, Rao contour, expansion trades |
| Regen cooling | `physics/regen/` | Bartz, channel sizing, T_wall |
| Injector | `physics/injector/` | Element sizing, stability |
| Structural | `physics/structural/` | Hoop stress, life, safety factors |
| Geometry kernel | `geometry/kernel/` | SDF primitives, mesh export |
| Chamber geometry | `geometry/chamber/` | Chamber + cooling channel surfaces |
| Nozzle geometry | `geometry/nozzle/` | Contour generation |
| Injector geometry | `geometry/injector/` | Pintle, manifold, orifices |

---

## Authoritative References

Keep these books/papers handy. When in doubt, cite one of these.

- **Huzel & Huang** — *Modern Engineering for Design of Liquid-Propellant
  Rocket Engines* (NASA SP-125). The canonical sizing reference.
- **Sutton & Biblarz** — *Rocket Propulsion Elements* (9th ed). General
  reference for everything.
- **Bartz (1957)** — NACA TN-2974, gas-side heat transfer.
- **NASA SP-8087** — Liquid Rocket Engine Fluid-Cooled Combustion Chambers.
- **NASA SP-8089** — Liquid Rocket Engine Injectors.
- **CEA / NASA RP-1311** — Gordon & McBride, equilibrium thermochemistry.
- **Pizzarelli et al.** — Transcritical methane heat transfer (modern regen).
- **Dressler & Bauer (TRW)** — Pintle injector design.

---

## SDF Surface Tangency — Design Rule

When two components share an **exact boundary plane** (both SDFs = 0 at the
same z), `smooth_union(0, 0, k)` produces only `−k/4` of solid depth at the
seam.  With the default `k = 1.5 mm` this is `−0.375 mm` — below the 0.4 mm
voxel pitch.  Marching cubes renders it as an invisible knife-edge seam;
Fusion 360 shows the part as floating.

**Fix:** make one component overlap the other by at least 3× voxel pitch
(≥ 1.2 mm; 2.5 mm used in practice for a 6× margin).  Do **not** increase
`k` to fix this — a larger blend radius blurs every adjacent surface.

**Applied:** `LoxDome` extends its shell 8 mm downstream past `x_chamber_start`
via `_LOX_DOME_TO_JACKET_OVERLAP = 8 mm` (= `_FACE_THICKNESS`).  This reaches
the chamber jacket's upstream face (which is trimmed 8 mm by
`engine_assembly_sdf._L_trimmed`), closing the 5.5 mm axial void at
r > r_chamber that was left by the earlier 2.5 mm floor-only fix.

**Test guards:**
- `test_dome_face_plate_overlap_visible` asserts SDF ≤ −1.5 mm at
  z = x_chamber_start + 5 mm (inside the dome floor solid zone).  Probe must be
  in the jointly-solid region (z > z_cav_hi = z_hi_extended − wall_t); probing
  in the cavity zone (z < z_cav_hi) fails because the dome SDF is positive there.
- `test_dome_outer_wall_connects_to_jacket` asserts 16 azimuthal points at
  r = r_outer − 1 mm, z = x_chamber_start + 4 mm are all solid.  This is the
  exact coordinate band that Fusion 360 showed as void before the 8 mm fix.

**Interface continuity lesson:** Tests must sample at the *visible failure
coordinates* (radial band, axial range) not just near the seam.  When debugging
"X floats in CAD," identify the gap's r and z range from the SDF diagnostic,
then assert SDF < 0 specifically there.  The 2.5 mm floor-lip fix (a68bafb)
passed all inner-band tests while the outer-band void remained visible because
no test sampled at r > r_chamber.  See commits a68bafb and this commit.

---

## What Goes Where (Quick Reference for Subagents)

When in doubt about which folder a new file belongs in:

- A function that takes thermo state and returns thermo state → `physics/thermochem/`
- A function that takes geometry and returns geometry → `geometry/`
- A function that takes geometry + thermo and returns wall temperatures → `physics/regen/`
- A standalone "let me check this number" script → `calcs/`
- A pytest file → `verification/`
- A reusable helper used by multiple subsystems → `physics/util/`
