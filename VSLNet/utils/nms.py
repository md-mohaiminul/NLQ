# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================
import torch

def nms(segments, scores, overlap=0.5, top_k=1000):
    left = segments[:, 0]
    right = segments[:, 1]
    #scores = segments[:, 2]
    keep = scores.new_zeros(scores.size(0)).long()
    area = right - left
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        l = torch.index_select(left, 0, idx)
        r = torch.index_select(right, 0, idx)
        l = torch.max(l, left[i])
        r = torch.min(r, right[i])
        # l = torch.clamp(l, max=left[i])
        # r = torch.clamp(r, min=right[i])
        inter = torch.clamp(r - l, min=0.0)
        rem_areas = torch.index_select(area, 0, idx)
        union = rem_areas - inter + area[i]
        IoU = inter / union
        idx = idx[IoU < overlap]
    return segments[keep], count

def softnms_v2(segments, tscore, sigma=0.5, top_k=1000, score_threshold=0.001):
    segments = segments.cpu()
    tstart = segments[:, 0]
    tend = segments[:, 1]
    #tscore = segments[:, 2]
    done_mask = tscore < -1    # set all to False
    undone_mask = tscore >= score_threshold
    while undone_mask.sum() > 1 and done_mask.sum() < top_k:
        idx = tscore[undone_mask].argmax()
        idx = undone_mask.nonzero()[idx].item()
        undone_mask[idx] = False
        done_mask[idx] = True
        top_start = tstart[idx]
        top_end = tend[idx]
        _tstart = tstart[undone_mask]
        _tend = tend[undone_mask]
        tt1 = _tstart.clamp(min=top_start)
        tt2 = _tend.clamp(max=top_end)
        intersection = torch.clamp(tt2 - tt1, min=0)
        duration = _tend - _tstart
        tmp_width = torch.clamp(top_end - top_start, min=1e-5)
        iou = intersection / (tmp_width + duration - intersection)
        scales = torch.exp(-iou**2 / sigma)
        tscore[undone_mask] *= scales
        undone_mask[tscore < score_threshold] = False
    count = done_mask.sum()
    segments = torch.stack([tstart[done_mask], tend[done_mask], tscore[done_mask]], -1)
    return segments, count