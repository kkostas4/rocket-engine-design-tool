import warnings
warnings.warn(
    "physics.heat_exchanger is deprecated; use noyron.kernel.heat_exchanger",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.heat_exchanger import *  # noqa: F401, F403, E402
from noyron.kernel.heat_exchanger import __getattr__  # noqa: F401, E402
