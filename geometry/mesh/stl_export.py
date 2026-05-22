import warnings
warnings.warn(
    "geometry.mesh.stl_export is deprecated; use noyron.kernel.geometry.mesh.stl_export",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.mesh.stl_export import *  # noqa: F401, F403, E402
