import warnings
warnings.warn(
    "physics.heat_exchanger.rocket_regen is deprecated; use physics.regen.rocket_regen",
    DeprecationWarning,
    stacklevel=2,
)
from physics.regen.rocket_regen import *  # noqa: F401, F403, E402
