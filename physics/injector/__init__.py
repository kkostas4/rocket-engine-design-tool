import warnings
warnings.warn(
    "physics.injector is deprecated; use noyron.rp.injector",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.injector import *  # noqa: F401, F403, E402
