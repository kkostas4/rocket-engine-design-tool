import warnings
warnings.warn(
    "physics.regen.bartz is deprecated; use noyron.rp.regen.bartz",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen.bartz import *  # noqa: F401, F403, E402
