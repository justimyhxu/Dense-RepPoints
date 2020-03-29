import cv2
import mmcv
import numpy as np
import torch

from mmdet.core.bbox import assign_and_sample, build_assigner, PseudoSampler
from mmdet.core.utils import multi_apply


def dense_reppoints_target(proposals_list,
                           proposals_pts_list,
                           valid_flag_list,
                           gt_bboxes_list,
                           gt_masks_list,
                           img_metas,
                           cfg,
                           gt_bboxes_ignore_list=None,
                           gt_labels_list=None,
                           label_channels=1,
                           sampling=True,
                           unmap_outputs=True,
                           num_pts=49, ):
    """Compute refinement and classification targets for points.

        Args:
            proposals_list(list(list)): Multi level bouding box of each image
            proposals_pts_list (list(list)): Multi level points of each image.
            valid_flag_list (list(list)): Multi level valid flags of each image.
            gt_bboxes_list (list(Tensor)): Ground truth bboxes of each image.
            img_metas (list(dict)): Meta info of each image.
            cfg (dict): Train sample configs.
            num_pts(int) Number of point sets

        Returns:
            tuple
    """
    num_imgs = len(img_metas)
    assert len(proposals_list) == len(valid_flag_list) == len(proposals_pts_list) == num_imgs

    # points number of multi levels
    num_level_proposals = [points.size(0) for points in proposals_list[0]]
    num_level_proposals_list = [num_level_proposals] * num_imgs
    # concat all level points and flags to a single tensor
    for i in range(num_imgs):
        assert len(proposals_list[i]) == len(valid_flag_list[i])
        proposals_list[i] = torch.cat(proposals_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])
        proposals_pts_list[i] = torch.cat(proposals_pts_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]
    (all_labels, all_label_weights, all_bbox_gt, all_mask_gt_index, all_mask_gt, all_mask_gt_label, all_proposals,
     all_proposal_weights, pos_inds_list, neg_inds_list) = multi_apply(
        dense_reppoints_target_sinle,
        proposals_list,
        proposals_pts_list,
        num_level_proposals_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_masks_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        num_pts=num_pts,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)
    # no valid points
    if any([labels is None for labels in all_labels]):
        return None
    # sampled points of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    labels_list = images_to_levels(all_labels, num_level_proposals, keep_dim=True)
    label_weights_list = images_to_levels(all_label_weights, num_level_proposals, keep_dim=True)
    bbox_gt_list = images_to_levels(all_bbox_gt, num_level_proposals, keep_dim=True)
    proposals_list = images_to_levels(all_proposals, num_level_proposals, keep_dim=True)
    proposal_weights_list = images_to_levels(all_proposal_weights, num_level_proposals, keep_dim=True)
    mask_gt_index_list = images_to_levels(all_mask_gt_index, num_level_proposals, keep_dim=True)
    mask_gt_list = mask_to_levels(all_mask_gt, mask_gt_index_list)
    mask_gt_label_list = mask_to_levels(all_mask_gt_label, mask_gt_index_list)

    return (labels_list, label_weights_list, bbox_gt_list, mask_gt_list, mask_gt_label_list, proposals_list,
            proposal_weights_list, num_total_pos, num_total_neg)


def images_to_levels(target, num_level_grids, keep_dim=False):
    """
    Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_grids:
        end = start + n
        if not keep_dim:
            level_targets.append(target[:, start:end].squeeze(0))
        else:
            level_targets.append(target[:, start:end])
        start = end
    return level_targets


def mask_to_levels(target, mask_index_list):
    """
    Convert target by mask_index_list
    """
    target_gt_list = []
    for lvl in range(len(mask_index_list)):
        mask_gt_lvl_list = []
        for i in range(mask_index_list[lvl].shape[0]):
            index = mask_index_list[lvl][i]
            index = index[index > 0]
            mask_gt_lvl = target[i][index - 1]
            mask_gt_lvl_list.append(mask_gt_lvl)
        target_gt_list.append(mask_gt_lvl_list)
    return target_gt_list


def dense_reppoints_target_sinle(flat_proposals,
                                 flat_proposals_pts,
                                 num_level_proposals,
                                 valid_flags,
                                 gt_bboxes,
                                 gt_masks,
                                 gt_bboxes_ignore,
                                 gt_labels,
                                 cfg,
                                 label_channels=1,
                                 sampling=True,
                                 unmap_outputs=True,
                                 num_pts=49):
    inside_flags = valid_flags
    num_level_proposals_inside = get_num_level_proposals_inside(num_level_proposals, inside_flags)

    if not inside_flags.any():
        return (None,) * 8
    # assign gt and sample points
    proposals = flat_proposals[inside_flags, :]
    proposals_pts = flat_proposals_pts[inside_flags, :]
    if sampling:
        assign_result, sampling_result = assign_and_sample(
            proposals, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        if cfg.assigner.type != "ATSSAssigner":
            assign_result = bbox_assigner.assign(proposals, gt_bboxes, None, gt_labels)
        else:
            assign_result = bbox_assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, None, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, proposals,
                                              gt_bboxes)

    gt_ind = sampling_result.pos_assigned_gt_inds.cpu().numpy()
    sample_func = cfg.get('sample_func', 'distance_sample_pts')
    gt_pts_numpy = eval(sample_func)(gt_bboxes, gt_masks, cfg, num_pts)

    pts_label_list = []
    proposals_pos_pts = proposals_pts[sampling_result.pos_inds, :].detach().cpu().numpy().round().astype(np.long)
    for i in range(len(gt_ind)):
        gt_mask = gt_masks[gt_ind[i]]
        h, w = gt_mask.shape
        pts_long = proposals_pos_pts[i]
        _pts_label = gt_mask[pts_long[1::2].clip(0, h - 1), pts_long[0::2].clip(0, w - 1)]
        pts_label_list.append(_pts_label)
    del proposals_pos_pts

    if len(gt_ind) != 0:
        gt_pts = gt_bboxes.new_tensor(gt_pts_numpy)
        pos_gt_pts = gt_pts[gt_ind]
        pts_label = np.stack(pts_label_list, 0)
        pos_gt_pts_label = gt_bboxes.new_tensor(pts_label)
    else:
        pos_gt_pts = None
        pos_gt_pts_label = None

    num_valid_proposals = proposals.shape[0]
    bbox_gt = proposals.new_zeros([num_valid_proposals, 4])
    mask_gt = proposals.new_zeros([0, num_pts * 2])
    mask_gt_label = proposals.new_zeros([0, num_pts]).long()
    mask_gt_index = proposals.new_zeros([num_valid_proposals, ], dtype=torch.long)
    pos_proposals = torch.zeros_like(proposals)
    proposals_weights = proposals.new_zeros([num_valid_proposals, 4])
    labels = proposals.new_zeros(num_valid_proposals, dtype=torch.long)
    label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        pos_gt_bboxes = sampling_result.pos_gt_bboxes
        bbox_gt[pos_inds, :] = pos_gt_bboxes
        if pos_gt_pts is not None:
            mask_gt = pos_gt_pts.type(bbox_gt.type())
            mask_gt_index[pos_inds] = torch.arange(len(pos_inds)).long().cuda() + 1
        if pos_gt_pts_label is not None:
            mask_gt_label = pos_gt_pts_label.long()
        pos_proposals[pos_inds, :] = proposals[pos_inds, :]
        proposals_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    # map up to original set of grids
    if unmap_outputs:
        num_total_proposals = flat_proposals.size(0)
        labels = unmap(labels, num_total_proposals, inside_flags)
        label_weights = unmap(label_weights, num_total_proposals, inside_flags)
        bbox_gt = unmap(bbox_gt, num_total_proposals, inside_flags)
        mask_gt_index = unmap(mask_gt_index, num_total_proposals, inside_flags)
        pos_proposals = unmap(pos_proposals, num_total_proposals, inside_flags)
        proposals_weights = unmap(proposals_weights, num_total_proposals, inside_flags)

    return (labels, label_weights, bbox_gt, mask_gt_index, mask_gt, mask_gt_label, pos_proposals, proposals_weights,
            pos_inds, neg_inds)


def unmap(data, count, inds, fill=0):
    """
    Unmap a subset of item (data) back to the original set of items (of size count)
    """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def get_num_level_proposals_inside(num_level_proposals, inside_flags):
    """
    Get number of proposal in different level
    """
    split_inside_flags = torch.split(inside_flags, num_level_proposals)
    num_level_proposals_inside = [
        int(flags.sum()) for flags in split_inside_flags
    ]
    return num_level_proposals_inside


def mask_to_poly(mask):
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:
            polygons.append(contour)
    return polygons


def distance_sample_pts(gt_bboxes, gt_masks, cfg, num_pts):
    """
    Sample pts based on distance transformation map.

    Args:
        gt_bboxes(list(Tensor)): groud-truth bounding box
        gt_masks(list(Mask)): ground-truth mask
        cfg(dict): sampling config
        num_pts(int): number of points

    Returns:
        numpy: the sampling points based on distance transform map
    """
    dist_sample_thr = cfg.get('dist_sample_thr', 3)
    pts_list = []
    pts_label_list = []
    for i in range(len(gt_bboxes)):
        x1, y1, x2, y2 = gt_bboxes[i].cpu().numpy().astype(np.int32)
        w = np.maximum(x2 - x1 + 1, 1)
        h = np.maximum(y2 - y1 + 1, 1)
        mask = mmcv.imresize(gt_masks[i][y1:y1 + h, x1:x1 + w],
                             (cfg.get('mask_size', 56), cfg.get('mask_size', 56)))
        polygons = mask_to_poly(mask)
        distance_map = np.ones(mask.shape).astype(np.uint8)
        for poly in polygons:
            poly = np.array(poly).astype(np.int)
            for j in range(len(poly) // 2):
                x_0, y_0 = poly[2 * j:2 * j + 2]
                if j == len(poly) // 2 - 1:
                    x_1, y_1 = poly[0:2]
                else:
                    x_1, y_1 = poly[2 * j + 2:2 * j + 4]
                cv2.line(distance_map, (x_0, y_0), (x_1, y_1), 0, thickness=2)
        roi_dist_map = cv2.distanceTransform(distance_map, cv2.DIST_L2, 3)
        con_index = np.stack(np.nonzero(roi_dist_map == 0)[::-1], axis=-1)
        roi_dist_map[roi_dist_map == 0] = 1
        roi_dist_map[roi_dist_map > dist_sample_thr] = 0

        index_y, index_x = np.nonzero(roi_dist_map > 0)
        index = np.stack([index_x, index_y], axis=-1)
        _len = index.shape[0]
        if len(con_index) == 0:
            pts = np.zeros([2 * num_pts])
        else:
            repeat = num_pts // _len
            mod = num_pts % _len
            perm = np.random.choice(_len, mod, replace=False)
            draw = [index.copy() for i in range(repeat)]
            draw.append(index[perm])
            draw = np.concatenate(draw, 0)
            draw = np.random.permutation(draw)
            draw = draw + np.random.rand(*draw.shape)
            x_scale = float(w) / cfg.get('mask_size', 56)
            y_scale = float(h) / cfg.get('mask_size', 56)
            draw[:, 0] = draw[:, 0] * x_scale + x1
            draw[:, 1] = draw[:, 1] * y_scale + y1
            pts = draw.reshape(2 * num_pts)

        pts_list.append(pts)
        pts_long = pts.astype(np.long)
        pts_label = gt_masks[i][pts_long[1::2], pts_long[0::2]]
        pts_label_list.append(pts_label)
    pts_list = np.stack(pts_list, 0)
    return pts_list
