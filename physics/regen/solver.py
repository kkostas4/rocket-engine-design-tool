import warnings
warnings.warn(
    "physics.regen.solver is deprecated; use noyron.rp.regen.solver",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.regen.solver import *  # noqa: F401, F403, E402
