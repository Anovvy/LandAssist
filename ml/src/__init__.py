"""LandAssist ML & Geospatial Post-Processing Pipeline."""

__version__ = "0.1.0"

from .model import (
    CLASS_MAPPING,
    ID2LABEL,
    LABEL2ID,
    WeightedCEDiceLoss,
    build_segformer_64ch,
    load_versioned_checkpoint,
    save_versioned_checkpoint,
)
from .infer_tangerang import (
    build_2d_hann_window,
    run_sliding_window_inference,
)
from .vectorize import (
    chaikin_smooth_ring,
    prob_map_to_osm_polygons,
    smooth_and_simplify_polygon,
)

__all__ = [
    "__version__",
    "CLASS_MAPPING",
    "ID2LABEL",
    "LABEL2ID",
    "WeightedCEDiceLoss",
    "build_segformer_64ch",
    "save_versioned_checkpoint",
    "load_versioned_checkpoint",
    "build_2d_hann_window",
    "run_sliding_window_inference",
    "chaikin_smooth_ring",
    "smooth_and_simplify_polygon",
    "prob_map_to_osm_polygons",
]
