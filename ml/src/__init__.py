"""LandAssist ML & Geospatial Post-Processing Pipeline."""

__version__ = "0.1.0"

from .infer_tangerang import (
    build_2d_hann_window,
    run_sliding_window_inference,
)
from .model import (
    CLASS_MAPPING,
    ID2LABEL,
    LABEL2ID,
    WeightedCEDiceLoss,
    build_segformer_64ch,
    load_versioned_checkpoint,
    save_versioned_checkpoint,
)
from .vectorize import (
    chaikin_smooth_ring,
    prob_map_to_osm_polygons,
    smooth_and_simplify_polygon,
)

__all__ = [
    "CLASS_MAPPING",
    "ID2LABEL",
    "LABEL2ID",
    "WeightedCEDiceLoss",
    "__version__",
    "build_2d_hann_window",
    "build_segformer_64ch",
    "chaikin_smooth_ring",
    "load_versioned_checkpoint",
    "prob_map_to_osm_polygons",
    "run_sliding_window_inference",
    "save_versioned_checkpoint",
    "smooth_and_simplify_polygon",
]
