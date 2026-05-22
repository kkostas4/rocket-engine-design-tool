import warnings
warnings.warn(
    "physics.structural is deprecated; use noyron.rp.structural",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.structural import *  # noqa: F401, F403, E402
