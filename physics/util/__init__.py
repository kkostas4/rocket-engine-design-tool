import warnings
warnings.warn(
    "physics.util is deprecated; use noyron.kernel.util",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.kernel.util import *  # noqa: F401, F403, E402
