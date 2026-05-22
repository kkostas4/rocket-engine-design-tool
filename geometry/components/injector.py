import warnings
warnings.warn(
    "geometry.components.injector is deprecated; use noyron.rp.geometry.injector",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.geometry.injector import *  # noqa: F401, F403, E402
