import warnings
warnings.warn(
    "physics.injector.stability is deprecated; use noyron.rp.injector.stability",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.injector.stability import *  # noqa: F401, F403, E402
