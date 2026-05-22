import warnings
warnings.warn(
    "physics.injector.discharge is deprecated; use noyron.rp.injector.discharge",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.injector.discharge import *  # noqa: F401, F403, E402
