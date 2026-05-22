import warnings
warnings.warn(
    "physics.heat_exchanger.solver is deprecated; use noyron.kernel.heat_exchanger.solver",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.heat_exchanger.solver import *  # noqa: F401, F403, E402
