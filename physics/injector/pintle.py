import warnings
warnings.warn(
    "physics.injector.pintle is deprecated; use noyron.rp.injector.pintle",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.injector.pintle import *  # noqa: F401, F403, E402
