import warnings
warnings.warn(
    "geometry.components is deprecated; use noyron.rp.geometry",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.geometry import *  # noqa: F401, F403, E402
