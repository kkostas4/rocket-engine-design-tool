import warnings
warnings.warn(
    "geometry.components.cooling_channels is deprecated; use noyron.rp.geometry.cooling_channels",
    DeprecationWarning,
    stacklevel=2,
)
from noyron.rp.geometry.cooling_channels import *  # noqa: F401, F403, E402
