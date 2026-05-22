import warnings
warnings.warn(
    "geometry.mesh.extractor is deprecated; use noyron.kernel.geometry.mesh.extractor",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.mesh.extractor import *  # noqa: F401, F403, E402
