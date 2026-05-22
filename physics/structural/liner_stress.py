import warnings
warnings.warn(
    "physics.structural.liner_stress is deprecated; use noyron.rp.structural.liner_stress",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural.liner_stress import *  # noqa: F401, F403, E402
