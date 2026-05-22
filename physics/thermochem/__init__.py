import warnings
warnings.warn(
    "physics.thermochem is deprecated; use noyron.rp.thermochem",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.thermochem import *  # noqa: F401, F403, E402
