import warnings
warnings.warn(
    "physics.util.units is deprecated; use noyron.kernel.util.units",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.util.units import *  # noqa: F401, F403, E402
