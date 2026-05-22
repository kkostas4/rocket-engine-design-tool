"""noyron.engine — backward-compat shim; real module is noyron.rp.engine."""
import warnings
warnings.warn(
    "noyron.engine is deprecated; use noyron.rp.engine",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.engine import *  # noqa: F401, F403, E402
from noyron.rp.engine import Engine, EngineDesign, EngineRequirements  # noqa: F401, E402
