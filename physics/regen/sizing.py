import warnings
warnings.warn(
    "physics.regen.sizing is deprecated; use noyron.rp.regen.sizing",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen.sizing import *  # noqa: F401, F403, E402
