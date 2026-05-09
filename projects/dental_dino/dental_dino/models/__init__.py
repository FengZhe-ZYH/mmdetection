# Register custom detectors / modules when package is imported.
from . import data_preprocessors  # noqa: F401
from .detectors import dino_tooth_prior  # noqa: F401
