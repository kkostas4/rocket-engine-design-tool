import warnings
warnings.warn(
    "physics.structural.materials is deprecated; use noyron.rp.structural.materials",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural.materials import *  # noqa: F401, F403, E402
