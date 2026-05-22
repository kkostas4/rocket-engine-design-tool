import warnings
warnings.warn(
    "physics.thermochem.cea is deprecated; use noyron.rp.thermochem.cea",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.thermochem.cea import *  # noqa: F401, F403, E402
