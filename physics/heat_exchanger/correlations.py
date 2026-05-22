import warnings
warnings.warn(
    "physics.heat_exchanger.correlations is deprecated; use noyron.kernel.heat_exchanger.correlations",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.heat_exchanger.correlations import *  # noqa: F401, F403, E402
