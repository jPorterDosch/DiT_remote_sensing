from __future__ import annotations
import os
import copy
import glob
import queue
import gc
import cv2
import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from registry import register_task


def _norm_mask(mask: torch.Tensor) -> torch.Tensor:
    c, h, w = mask.size()
    for cnt in range(c):
        mask_cnt = mask[cnt, :, :]
        if mask_cnt.max() > 0:
            mask_cnt = (mask_cnt - mask_cnt.min())
            mask_cnt = mask_cnt / mask_cnt.max()
            mask[cnt, :, :] = mask_cnt
    return mask


def _restrict_neighborhood(h: int, w: int, size_mask_neighborhood: int) -> torch.Tensor:
    # We restrict the set of source nodes considered to a spatial neighborhood of the query node (i.e. ``local attention'')
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1
    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)


def _extract_feature(cfg, model, frame: torch.Tensor, ori_h: int, ori_w: int, return_h_w: bool = False):
    """Extract one frame feature everytime."""
    with torch.no_grad():
        feat      = model.extract(
            frame,
            timestep=cfg.t,
            block_idx=cfg.k,
            ensemble_size=cfg.model.ensemble_size,
        )  # 1, C, H, W
        feat      = feat.squeeze(0)                # C, H, W
        _c, h, w  = feat.shape
        feat      = torch.permute(feat, (1, 2, 0)) # h, w, c
        feat      = feat.view(h * w, _c)           # hw, c
        if return_h_w:
            return feat, h, w
        return feat


def _label_propagation(cfg, model, frame_tar, list_frame_feats, list_segs, ori_h, ori_w, mask_neighborhood=None):
    """
    propagate segs of frames in list_frames to frame_tar
    """
    gc.collect()
    torch.cuda.empty_cache()

    ## we only need to extract feature of the target frame
    feat_tar, h, w = _extract_feature(cfg, model, frame_tar, ori_h, ori_w, return_h_w=True)

    gc.collect()
    torch.cuda.empty_cache()

    return_feat_tar = feat_tar.T # dim x h*w

    ncontext     = len(list_frame_feats)
    feat_sources = torch.stack(list_frame_feats) # nmb_context x dim x h*w

    feat_tar     = F.normalize(feat_tar, dim=1, p=2)
    feat_sources = F.normalize(feat_sources, dim=1, p=2)

    feat_tar = feat_tar.unsqueeze(0).repeat(ncontext, 1, 1)
    aff = torch.exp(torch.bmm(feat_tar, feat_sources) / cfg.temperature) # nmb_context x h*w (tar:  query) x h*w (source:  keys)

    if cfg.size_mask_neighborhood > 0:
        if mask_neighborhood is None:
            mask_neighborhood = _restrict_neighborhood(h, w, cfg.size_mask_neighborhood)
            mask_neighborhood = mask_neighborhood.unsqueeze(0).repeat(ncontext, 1, 1)
        aff *= mask_neighborhood

    aff = aff.float().transpose(2, 1).reshape(-1, h * w) # nmb_context*h*w (source:  keys) x h*w (tar:  queries)
    tk_val, _ = torch.topk(aff, dim=0, k=cfg.topk)
    tk_val_min, _ = torch.min(tk_val, dim=0)
    aff[aff < tk_val_min] = 0

    aff = aff / torch.sum(aff, keepdim=True, axis=0)

    gc.collect()
    torch.cuda.empty_cache()

    list_segs = [s.cuda() for s in list_segs]
    segs = torch.cat(list_segs)
    nmb_context, C, h, w = segs.shape
    segs    = segs.reshape(nmb_context, C, -1).transpose(2, 1).reshape(-1, C).T # C x nmb_context*h*w
    seg_tar = torch.mm(segs, aff)
    seg_tar = seg_tar.reshape(1, C, h, w)

    return seg_tar, return_feat_tar, mask_neighborhood


def _imwrite_indexed(filename: str, array, color_palette) -> None:
    """ Save indexed png for DAVIS."""
    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")
    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format="PNG")


def _to_one_hot(y_tensor, n_dims=None):
    """
    Take integer y (tensor or variable) with n dims &
    convert it to 1-hot representation with n+1 dims.
    """
    if n_dims is None:
        n_dims = int(y_tensor.max() + 1)
    _, h, w   = y_tensor.size()
    y_tensor  = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims    = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(h, w, n_dims)
    return y_one_hot.permute(2, 0, 1).unsqueeze(0)


def _read_frame_list(video_dir: str) -> list[str]:
    frame_list = [img for img in glob.glob(os.path.join(video_dir, "*.jpg"))]
    return sorted(frame_list)


def _color_normalize(x, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    for t, m, s in zip(x, mean, std):
        t.sub_(m)
        t.div_(s)
    return x


def _read_frame(frame_dir: str, scale_size: list[int] = [960]):
    """read a single frame & preprocess"""
    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if ori_h > ori_w:
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 32) * 32)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 32) * 32)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = _color_normalize(img)
    return img, ori_h, ori_w


def _read_seg(seg_dir: str, scale_factor: int, scale_size: list[int] = [960]):
    seg    = Image.open(seg_dir)
    _w, _h = seg.size  # note PIL.Image.Image's size is (w, h)
    if len(scale_size) == 1:
        if _w > _h:
            _th = scale_size[0]
            _tw = (_th * _w) / _h
            _tw = int((_tw // 32) * 32)
        else:
            _tw = scale_size[0]
            _th = (_tw * _h) / _w
            _th = int((_th // 32) * 32)
    else:
        _th = scale_size[1]
        _tw = scale_size[0]
    small_seg = np.array(seg.resize((_tw // scale_factor, _th // scale_factor), 0))
    small_seg = torch.from_numpy(small_seg.copy()).contiguous().float().unsqueeze(0)
    return _to_one_hot(small_seg), np.asarray(seg)


@torch.no_grad()
def _eval_video_tracking_davis(cfg, model, scale_factor, frame_list, video_dir, first_seg, seg_ori, color_palette):
    """Evaluate tracking on a video given first frame & segmentation"""
    video_folder = os.path.join(cfg.output_dir, video_dir.split("/")[-1])
    os.makedirs(video_folder, exist_ok=True)

    img_size = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size

    # The queue stores the n preceeding frames
    que = queue.Queue(cfg.n_last_frames)

    # first frame
    frame1, ori_h, ori_w = _read_frame(frame_list[0], scale_size=[img_size])
    # extract first frame feature
    frame1_feat = _extract_feature(cfg, model, frame1, ori_h, ori_w).T  # dim x h*w

    # saving first segmentation
    _imwrite_indexed(os.path.join(video_folder, "00000.png"), seg_ori, color_palette)
    mask_neighborhood = None

    for cnt in tqdm(range(1, len(frame_list))):
        frame_tar = _read_frame(frame_list[cnt], scale_size=[img_size])[0]

        # we use the first segmentation and the n previous ones
        used_frame_feats = [frame1_feat] + [pair[0] for pair in list(que.queue)]
        used_segs        = [first_seg]   + [pair[1] for pair in list(que.queue)]

        frame_tar_avg, feat_tar, mask_neighborhood = _label_propagation(
            cfg, model, frame_tar, used_frame_feats, used_segs, ori_h, ori_w, mask_neighborhood
        )

        # pop out oldest frame if neccessary
        if que.qsize() == cfg.n_last_frames:
            que.get()
        # push current results into queue
        seg = copy.deepcopy(frame_tar_avg)
        que.put([feat_tar, seg])

        # upsampling & argmax
        frame_tar_avg = F.interpolate(
            frame_tar_avg, scale_factor=scale_factor,
            mode="bilinear", align_corners=False, recompute_scale_factor=False,
        )[0]
        frame_tar_avg = _norm_mask(frame_tar_avg)
        _, frame_tar_seg = torch.max(frame_tar_avg, dim=0)

        # saving to disk
        frame_tar_seg = np.array(frame_tar_seg.squeeze().cpu(), dtype=np.uint8)
        frame_tar_seg = np.array(Image.fromarray(frame_tar_seg).resize((ori_w, ori_h), 0))
        frame_nm = frame_list[cnt].split("/")[-1].replace(".jpg", ".png")
        _imwrite_indexed(os.path.join(video_folder, frame_nm), frame_tar_seg, color_palette)


@register_task("segmentation")
class SegmentationTask:
    def run(self, cfg, model, dataset, results_dir: str) -> dict:
        data          = dataset.get_data(cfg)
        video_list    = data["video_list"]
        color_palette = data["color_palette"]
        scale_factor  = data["scale_factor"]

        n_last_frames = cfg.n_last_frames
        os.makedirs(cfg.output_dir, exist_ok=True)

        for i, video_name in enumerate(video_list):
            video_name = video_name.strip()

            if video_name == "shooting":
                if cfg.n_last_frames > 10:
                    cfg.n_last_frames = 10  # this can resolve the OOM issue
            else:
                cfg.n_last_frames = n_last_frames

            print(f"[{i}/{len(video_list)}] Begin to segmentate video {video_name}.")
            video_dir  = os.path.join(cfg.dataset.path, "JPEGImages/480p/", video_name)
            frame_list = _read_frame_list(video_dir)
            seg_path   = frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")
            img_size   = cfg.img_size[0] if isinstance(cfg.img_size, list) else cfg.img_size
            first_seg, seg_ori = _read_seg(seg_path, scale_factor, scale_size=[img_size])
            _eval_video_tracking_davis(
                cfg, model, scale_factor, frame_list, video_dir, first_seg, seg_ori, color_palette
            )

        return {}
