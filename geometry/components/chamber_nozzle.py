import warnings
warnings.warn(
    "geometry.components.chamber_nozzle is deprecated; use noyron.rp.geometry.chamber_nozzle",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.geometry.chamber_nozzle import *  # noqa: F401, F403, E402
