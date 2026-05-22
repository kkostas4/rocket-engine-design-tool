import warnings
warnings.warn(
    "geometry.sdf is deprecated; use noyron.kernel.geometry.sdf",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.sdf import *  # noqa: F401, F403, E402
