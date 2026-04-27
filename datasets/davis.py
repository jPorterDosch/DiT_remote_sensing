from __future__ import annotations
import os
import numpy as np
from registry import register_dataset


@register_dataset("davis")
class DAVISDataset:
    category_list: list[str] = []  # DAVIS uses "image" as the default prompt

    def __init__(self, cfg) -> None:
        pass

    def get_data(self, cfg) -> dict:
        color_palette: list[list[int]] = []
        with open("./1.txt") as f:
            for line in f:
                color_palette.append([int(i) for i in line.split('\n')[0].split(" ")])
        palette = np.asarray(color_palette, dtype=np.uint8).reshape(-1, 3)

        video_list = open(
            os.path.join(cfg.dataset.path, "ImageSets/2017/val.txt")
        ).readlines()

        index2factor = {
            0: 32, 1: 32, 2: 16, 3: 16, 4: 16, 5: 8, 6: 8, 7: 8, 8: 4,
            9: 4, 10: 4, 11: 2, 12: 2, 13: 2, 14: 1, 15: 1, 16: 1, 17: 1,
        }

        return {
            "video_list":    video_list,
            "color_palette": palette,
            "scale_factor":  index2factor[2],
        }
