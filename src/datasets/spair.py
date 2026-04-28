from __future__ import annotations
import os
import json
from registry import register_dataset


@register_dataset("spair")
class SPairDataset:
    def __init__(self, cfg) -> None:
        self.category_list: list[str] = os.listdir(os.path.join(cfg.dataset.path, "JPEGImages"))

    def get_data(self, cfg) -> dict:
        dataset_path = cfg.dataset.path
        test_path = "PairAnnotation/test"
        json_list = os.listdir(os.path.join(dataset_path, test_path))
        all_cats = self.category_list

        cat2json: dict[str, list[str]] = {cat: [j for j in json_list if cat in j] for cat in all_cats}

        cat2img: dict[str, list[str]] = {}
        for cat in all_cats:
            cat2img[cat] = []
            for json_path in cat2json[cat]:
                with open(os.path.join(dataset_path, test_path, json_path)) as f:
                    data = json.load(f)
                for key in ("src_imname", "trg_imname"):
                    if data[key] not in cat2img[cat]:
                        cat2img[cat].append(data[key])

        with open(cfg.captions_path) as f:
            captions = json.load(f)

        return {
            "dataset_path": dataset_path,
            "test_path": test_path,
            "all_cats": all_cats,
            "cat2json": cat2json,
            "cat2img": cat2img,
            "captions": captions,
        }
