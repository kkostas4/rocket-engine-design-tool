import warnings
warnings.warn(
    "physics.regen.rocket_regen is deprecated; use noyron.rp.regen.rocket_regen",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen.rocket_regen import *  # noqa: F401, F403, E402
