import warnings
warnings.warn(
    "physics.regen is deprecated; use noyron.rp.regen",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen import *  # noqa: F401, F403, E402
