import warnings
warnings.warn(
    "geometry.mesh is deprecated; use noyron.kernel.geometry.mesh",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.mesh import *  # noqa: F401, F403, E402
