import warnings
warnings.warn(
    "physics.structural.fatigue is deprecated; use noyron.rp.structural.fatigue",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural.fatigue import *  # noqa: F401, F403, E402
