"""
Central schema definitions for the HacoPita inference pipeline.

The list of FEATURE_COLUMNS_24 is the single source of truth for both
training and inference. The same list must also be exported as the
model artifact `model_features.json` so that the inference application
can validate what the deployed Azure ML model expects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

FEATURE_COLUMNS_24: List[str] = [
    "total_items",
    "bonsai",
    "other",
    "plastic_pots_trays",
    "single_flower_vase",
    "decorative_sand",
    "saucers_mats",
    "books",
    "suiban",
    "bonsai_seeds",
    "for_bonsai_classes",
    "bonsai_soil",
    "bonsai_tools",
    "bonsai_pots",
    "bonsai_decorations",
    "lucky_bag",
    "moss",
    "moss_bonsai",
    "chemicals_fertilizer",
    "wire",
    "decorative_stones",
    "accessories",
    "max_item_long_cm",
    "sum_item_volume_cm3",
]

TARGET_COLUMN = "box_id"
ID_COLUMN = "slip_number"

# Map historical column names to their canonical counterparts so that
# older CSV exports still work with the 24-feature model.
ALIASES: Dict[str, str] = {
    "others": "other",
    "water_basins": "suiban",
    "bonsai_class_items": "for_bonsai_classes",
    "bonsai_decor": "bonsai_decorations",
    "chemicals_fertilizers": "chemicals_fertilizer",
}

MODEL_FEATURES_PATH = Path(__file__).resolve().parent / "model_features.json"


def load_feature_columns(
    path: Path | None = None, *, strict: bool = True
) -> List[str]:
    """
    Load the ordered list of feature columns from `model_features.json`.

    When `strict` is True we verify the artifact matches FEATURE_COLUMNS_24.
    Missing files fall back to the constant so that local development can
    proceed even before a training job is executed.
    """

    candidate_path = Path(path) if path else MODEL_FEATURES_PATH
    if not candidate_path.exists():
        return FEATURE_COLUMNS_24

    with candidate_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    if not isinstance(loaded, list):
        raise ValueError(f"model_features.json must contain a list, got {type(loaded)!r}")

    if len(loaded) != len(FEATURE_COLUMNS_24):
        raise ValueError(
            "model_features.json does not contain 24 entries. "
            "Regenerate the artifact from the training job."
        )

    if strict and loaded != FEATURE_COLUMNS_24:
        raise ValueError(
            "model_features.json does not match FEATURE_COLUMNS_24. "
            "Please rebuild the model to keep schema in sync."
        )

    return loaded
