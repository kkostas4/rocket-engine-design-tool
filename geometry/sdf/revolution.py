import warnings
warnings.warn(
    "geometry.sdf.revolution is deprecated; use noyron.kernel.geometry.sdf.revolution",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.sdf.revolution import *  # noqa: F401, F403, E402
