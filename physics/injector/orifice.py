import warnings
warnings.warn(
    "physics.injector.orifice is deprecated; use noyron.rp.injector.orifice",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.injector.orifice import *  # noqa: F401, F403, E402
