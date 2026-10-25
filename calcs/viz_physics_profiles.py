import numpy as np
import matplotlib.pyplot as plt
from noyron import Engine, EngineRequirements, METHALOX
import yaml

with open('calcs/design_examples/methalox_5kN.yaml') as f:
    cfg = yaml.safe_load(f)
req = EngineRequirements(
    thrust=float(cfg['thrust']), chamber_pressure=float(cfg['pc']),
    mixture_ratio=float(cfg['mr']), propellant=METHALOX,
    expansion_ratio=float(cfg['epsilon']),
    T_wall_max=float(cfg['t_wall_max']),
    dP_coolant_budget=float(cfg['dp_coolant']),
    n_cycles_required=int(cfg['n_cycles']),
    pump_pressure_margin=float(cfg['pump_margin']),
    L_star=float(cfg['l_star']) if cfg.get('l_star') is not None else None,
    Ac_At=float(cfg.get('ac_at', 4.0)),
)
design = Engine(req).design()
s = design.regen_solution

x  = np.array(s['x']) * 1000
r  = np.array([design.nozzle.r(xi/1000) for xi in x]) * 1000
Tw = np.array(s['T_wall_gas'])
Tc = np.array(s['T_coolant'])
q  = np.array(s['q_dot']) / 1e6
hg = np.array(s['h_g']) / 1e3

fig, ax = plt.subplots(4, 1, figsize=(9, 11), sharex=True)
ax[0].plot(x, r); ax[0].fill_between(x, r, 0, alpha=.15); ax[0].set_ylabel("radius [mm]")
ax[0].set_title("The engine (top) and the physics along it")
ax[1].plot(x, q, 'r'); ax[1].set_ylabel("heat flux q'' [MW/m^2]")
ax[2].plot(x, hg, 'orange'); ax[2].set_ylabel("gas-side h_g [kW/m^2K]")
ax[3].plot(x, Tw, 'r', label='wall'); ax[3].plot(x, Tc, 'b', label='coolant')
ax[3].set_ylabel("temp [K]"); ax[3].legend(); ax[3].set_xlabel("axial position [mm]")
for a in ax: a.axvline(0, color='gray', ls='--')
plt.tight_layout(); plt.savefig('output/physics_profiles.png', dpi=120)
print("saved output/physics_profiles.png")
