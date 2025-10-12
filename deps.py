import os

from anchor_freezer import AnchorFreezer

FREEZER = AnchorFreezer(
    approach_ratio=float(os.getenv("ANCHOR_APPROACH_RATIO", "0.90")),
    freeze_secs=int(os.getenv("ANCHOR_FREEZE_SECS", "300")),
)

__all__ = ["FREEZER", "AnchorFreezer"]
