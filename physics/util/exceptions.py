import warnings
warnings.warn(
    "physics.util.exceptions is deprecated; use noyron.kernel.util.exceptions",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.util.exceptions import *  # noqa: F401, F403, E402
