import warnings
warnings.warn(
    "physics.nozzle is deprecated; use noyron.rp.nozzle",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.nozzle import *  # noqa: F401, F403, E402
