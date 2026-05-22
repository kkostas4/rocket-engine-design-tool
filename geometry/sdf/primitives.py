import warnings
warnings.warn(
    "geometry.sdf.primitives is deprecated; use noyron.kernel.geometry.sdf.primitives",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.sdf.primitives import *  # noqa: F401, F403, E402
