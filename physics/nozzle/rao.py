import warnings
warnings.warn(
    "physics.nozzle.rao is deprecated; use noyron.rp.nozzle.rao",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.nozzle.rao import *  # noqa: F401, F403, E402
