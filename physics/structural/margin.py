import warnings
warnings.warn(
    "physics.structural.margin is deprecated; use noyron.rp.structural.margin",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural.margin import *  # noqa: F401, F403, E402
