import warnings
warnings.warn(
    "physics.regen.coolant is deprecated; use noyron.rp.regen.coolant",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen.coolant import *  # noqa: F401, F403, E402
