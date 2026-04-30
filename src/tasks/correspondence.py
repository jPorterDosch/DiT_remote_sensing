from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import PILToTensor
from tqdm import tqdm

from registry import register_task


@register_task("correspondence")
class CorrespondenceTask:
    def run(self, cfg, model, dataset) -> dict:
        device = torch.device(cfg.device)

        data = dataset.get_data(cfg)
        dataset_path = data["dataset_path"]
        test_path = data["test_path"]
        all_cats = data["all_cats"]
        cat2json = data["cat2json"]
        cat2img = data["cat2img"]
        captions = data["captions"]

        #### feature extraction
        print("saving all test images' features...")
        os.makedirs(cfg.save_dir, exist_ok=True)

        for cat in tqdm(all_cats):
            feat_dict: dict[str, torch.Tensor] = {}
            for image_path in cat2img[cat]:
                img = Image.open(os.path.join(dataset_path, "JPEGImages", cat, image_path))

                ###preprocess
                image_arr = np.array(img)
                in_h, in_w = image_arr.shape[:2]
                scale = cfg.img_size[0] / max(in_h, in_w)
                H = int(round(in_h * scale / 16)) * 16  # 保证是16的倍数
                W = int(round(in_w * scale / 16)) * 16
                img = img.resize((W, H))
                img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2

                caption = captions[cat + image_path]
                feat = model.extract(
                    img_tensor,
                    timestep=cfg.t,
                    block_idx=cfg.k,
                    ensemble_size=cfg.model.ensemble_size,
                    caption=caption,
                    category=cat,
                )
                feat_dict[image_path] = feat.cpu()

            torch.save(feat_dict, os.path.join(cfg.save_dir, f"{cat}.pth"))

        #### evaluation
        total_pck = []
        all_correct = 0
        all_total = 0

        mean_image_sum = 0.0
        mean_point_sum = 0.0
        result = {"image": {}, "point": {}}

        print("Category numbers: %s" % len(all_cats))
        for cat in all_cats:
            cat_list = cat2json[cat]

            #### load data feature
            feat_dict = torch.load(os.path.join(cfg.save_dir, f"{cat}.pth"))

            cat_pck = []
            cat_correct = 0
            cat_total = 0

            for cat_idx, json_path in enumerate(tqdm(cat_list)):
                ##load image pair
                with open(os.path.join(dataset_path, test_path, json_path)) as f:
                    pair = json.load(f)

                src_img_size = pair["src_imsize"][:2][::-1]
                trg_img_size = pair["trg_imsize"][:2][::-1]

                # B,C,H,W
                src_ft = feat_dict[pair["src_imname"]].to(device)
                trg_ft = feat_dict[pair["trg_imname"]].to(device)
                B, C, H, W = src_ft.shape

                src_ft = src_ft.to(torch.float16)
                trg_ft = trg_ft.to(torch.float16)

                src_ft = nn.Upsample(size=src_img_size, mode="bilinear")(src_ft)
                trg_ft = nn.Upsample(size=trg_img_size, mode="bilinear")(trg_ft)

                h = trg_ft.shape[-2]
                w = trg_ft.shape[-1]

                trg_bndbox = pair["trg_bndbox"]
                threshold = max(
                    trg_bndbox[3] - trg_bndbox[1],
                    trg_bndbox[2] - trg_bndbox[0],
                )

                total = 0
                correct = 0
                src_list: list = []
                trg_list: list = []

                # print(len(pair['src_kps']))
                for idx in range(len(pair["src_kps"])):
                    total += 1
                    cat_total += 1
                    all_total += 1
                    src_point = pair["src_kps"][idx]
                    trg_point = pair["trg_kps"][idx]
                    src_list.append(src_point)
                    num_channel = src_ft.size(1)
                    src_vec = src_ft[0, :, src_point[1], src_point[0]].view(1, num_channel)  # 1, C
                    trg_vec = trg_ft.view(num_channel, -1).transpose(0, 1)  # HW, C
                    src_vec = F.normalize(src_vec).transpose(0, 1)  # c, 1
                    trg_vec = F.normalize(trg_vec)  # HW, c

                    cos_map = torch.mm(trg_vec, src_vec).view(h, w).cpu().numpy()  # H, W

                    max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)
                    trg_list.append([max_yx[1], max_yx[0]])
                    dist = ((max_yx[1] - trg_point[0]) ** 2 + (max_yx[0] - trg_point[1]) ** 2) ** 0.5
                    if (dist / threshold) <= 0.1:
                        correct += 1
                        cat_correct += 1
                        all_correct += 1

                cat_pck.append(correct / total)
                torch.cuda.empty_cache()

            total_pck.extend(cat_pck)
            mean_image_sum += np.mean(cat_pck) * 100
            mean_point_sum += cat_correct / cat_total * 100

            print(f"{cat} per image PCK@0.1: {np.mean(cat_pck) * 100:.2f}")
            print(f"{cat} per point PCK@0.1: {cat_correct / cat_total * 100:.2f}")

            result["image"][cat] = round(np.mean(cat_pck) * 100, 2)
            result["point"][cat] = round(cat_correct / cat_total * 100, 2)

        print(f"All per image PCK@0.1: {np.mean(total_pck) * 100:.2f}")
        print(f"All per point PCK@0.1: {all_correct / all_total * 100:.2f}")
        print(f"Mean per image PCK@0.1: {mean_image_sum / len(all_cats):.2f}")
        print(f"Mean per point PCK@0.1: {mean_point_sum / len(all_cats):.2f}")

        result["image"]["All"] = round(np.mean(total_pck) * 100, 2)
        result["point"]["All"] = round(all_correct / all_total * 100, 2)
        result["image"]["Mean"] = round(mean_image_sum / len(all_cats), 2)
        result["point"]["Mean"] = round(mean_point_sum / len(all_cats), 2)

        # print(result)
        out_path = os.path.join(
            cfg.save_dir,
            "t%s_b%s_e%s.json" % (cfg.t, cfg.k, cfg.model.ensemble_size),
        )
        with open(out_path, "w+") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

        return result
