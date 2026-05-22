import warnings
warnings.warn(
    "physics.structural.bilayer is deprecated; use noyron.rp.structural.bilayer",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural.bilayer import *  # noqa: F401, F403, E402
