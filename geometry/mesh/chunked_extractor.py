import warnings
warnings.warn(
    "geometry.mesh.chunked_extractor is deprecated; use noyron.kernel.geometry.mesh.chunked_extractor",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.geometry.mesh.chunked_extractor import *  # noqa: F401, F403, E402
