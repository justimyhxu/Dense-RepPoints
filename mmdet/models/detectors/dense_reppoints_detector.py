import cv2
import mmcv
import numpy as np
import scipy.interpolate
import torch
from mmcv.image import imread, imwrite
from pycocotools import mask as maskUtils

from mmdet.core import bbox2result, bbox_mapping_back, tensor2imgs, get_classes, multiclass_nms_pts
from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class DenseRepPointsDetector(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(DenseRepPointsDetector, self).__init__(backbone, neck, bbox_head, train_cfg,
                                                     test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=False)
        loss_inputs = outs + (gt_bboxes, gt_masks, gt_labels, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x, test=True)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        det_bboxes, det_points, det_pts_scores, det_cls = bbox_list[0]

        # cat pts_score to det_points for visualization
        det_points_reshape = det_points[:, :-1].reshape(det_points.shape[0], -1, 2)
        det_pts_scores_reshape = det_pts_scores[:, :-1].reshape(det_pts_scores.shape[0], -1, 1)
        det_pts_score_cat = torch.cat([det_points_reshape, det_pts_scores_reshape], dim=-1) \
            .reshape(det_points.shape[0], -1)
        det_pts_score_cls_cat = torch.cat([det_pts_score_cat, det_points[:, [-1]]], dim=-1)

        ori_shape = img_meta[0]['ori_shape']
        scale_factor = img_meta[0]['scale_factor']

        bbox_results = bbox2result(det_bboxes, det_cls, self.bbox_head.num_classes)
        pts_results = pts2result(det_pts_score_cls_cat, det_cls, self.bbox_head.num_classes)
        rle_results = self.get_seg_masks(det_pts_scores[:, :-1], det_points[:, :-1], det_bboxes, det_cls,
                                          self.test_cfg, ori_shape, scale_factor, rescale)
        # For visualization(rescale=False), we also return pts_results to show the points
        if not rescale:
            return (bbox_results, rle_results), pts_results
        else:
            return bbox_results, rle_results

    def get_seg_masks(self, pts_score, det_pts, det_bboxes, det_labels,
                      test_cfg, ori_shape, scale_factor, rescale=False):
        """
        Get segmentation masks from points and scores

        Args:
            pts_score (Tensor or ndarray): shape (n, num_pts)
            det_pts (Tensor): shape (n, num_pts*2)
            det_bboxes (Tensor): shape (n, 4)
            det_labels (Tensor): shape (n, 1)
            test_cfg (dict): rcnn testing config
            ori_shape: original image size
            scale_factor: scale factor for image
            rescale: whether rescale to original size
        Returns:
            list[list]: encoded masks
        """

        cls_segms = [[] for _ in range(self.bbox_head.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
        scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            im_pts = det_pts[i].clone()
            im_pts = im_pts.reshape(-1, 2)
            im_pts_score = pts_score[i]

            im_pts[:, 0] = (im_pts[:, 0] - bbox[0])
            im_pts[:, 1] = (im_pts[:, 1] - bbox[1])
            _h, _w = h, w
            corner_pts = im_pts.new_tensor([[0, 0], [_h - 1, 0], [0, _w - 1], [_w - 1, _h - 1]])
            corner_score = im_pts_score.new_tensor([0, 0, 0, 0])
            im_pts = torch.cat([im_pts, corner_pts], dim=0).cpu().numpy()
            im_pts_score = torch.cat([im_pts_score, corner_score], dim=0).cpu().numpy()
            grids = tuple(np.mgrid[0:_w:1, 0:_h:1])
            bbox_mask = scipy.interpolate.griddata(im_pts, im_pts_score, grids)
            bbox_mask = bbox_mask.transpose(1, 0)
            bbox_mask = mmcv.imresize(bbox_mask, (w, h))

            bbox_mask = bbox_mask.astype(np.float32)
            bbox_mask[np.isnan(bbox_mask)] = 0
            bbox_mask = (bbox_mask > test_cfg.get('pts_score_thr', 0.5)).astype(np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = maskUtils.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)
        return cls_segms

    def merge_aug_results(self, aug_bboxes, aug_scores, aug_points, aug_masks, img_metas):
        """
        Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            aug_points (list[Tensor]): shape (n, pts_num*#class)
            img_shapes (list[Tensor]): shape (3, ).
            rcnn_test_cfg (dict): rcnn test config.

        Returns:
            tuple: (bboxes, scores)
        """

        def points_mapping_back(points, img_shape, scale_factor, flip):
            flipped = points.clone()
            if flip:
                flipped[:, 0] = img_shape[1] - points[:, 0]
            flipped = flipped / scale_factor
            return flipped

        recovered_bboxes = []
        recovered_points = []
        for bboxes, points, img_info in zip(aug_bboxes, aug_points, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            points = points_mapping_back(points, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
            recovered_points.append(points)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        points = torch.cat(recovered_points, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            masks = torch.cat(aug_masks, dim=0)
            return bboxes, scores, points, masks

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        aug_masks = []
        aug_points = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            # TODO more flexible
            outs = self.bbox_head(x, test=True)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

            det_bbox, det_points, det_masks, det_labels = bbox_list[0]
            aug_bboxes.append(det_bbox)
            aug_scores.append(det_labels)
            aug_masks.append(det_masks)
            aug_points.append(det_points)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores, merged_points, merged_masks = self.merge_aug_results(
            aug_bboxes, aug_scores, aug_points, aug_masks, img_metas)

        cfg = self.test_cfg
        det_bboxes, det_pts, det_masks, det_labels = multiclass_nms_pts(
            merged_bboxes, merged_points, merged_scores, merged_masks, cfg.score_thr, cfg.nms, cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']

        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        pts_results = pts2result(det_pts, det_labels, self.bbox_head.num_classes)

        ori_shape = img_metas[0][0]['ori_shape']
        scale_factor = img_metas[0][0]['scale_factor']
        rle_results = self.get_seg_masks(det_masks[:, :-1], det_pts[:, :-1], _det_bboxes, det_labels, self.test_cfg,
                                         ori_shape, scale_factor, rescale)
        return (bbox_results, rle_results), pts_results

    def show_result(self,
                    data,
                    result,
                    img_norm_cfg,
                    dataset='coco',
                    score_thr=0.3,
                    out_file=None,
                    show=True,
                    draw_pts=True,
                    draw_mask=True,
                    **kwargs
                    ):
        bbox_seg_result, pts_result = result
        if isinstance(bbox_seg_result, tuple):
            bbox_result, segm_result = bbox_seg_result
        else:
            bbox_result, segm_result = bbox_seg_result, None
        img_tensor = data['img'][0]
        img_metas = data['img_meta'][0].data[0]
        imgs = tensor2imgs(img_tensor, **img_norm_cfg)
        assert len(imgs) == len(img_metas)

        if dataset is None:
            class_names = self.CLASSES
        elif isinstance(dataset, str):
            class_names = get_classes(dataset)
        elif isinstance(dataset, (list, tuple)):
            class_names = dataset
        else:
            raise TypeError(
                'dataset must be a valid dataset name or a sequence'
                ' of class names, not {}'.format(type(dataset)))

        for img, img_meta in zip(imgs, img_metas):
            h, w, _ = img_meta['img_shape']
            img_show = img[:h, :w, :]

            pts = np.vstack(pts_result)
            bboxes = np.vstack(bbox_result)
            # draw segmentation masks
            if segm_result is not None and draw_mask:
                segms = mmcv.concat_list(segm_result)
                inds = np.where(bboxes[:, -1] > score_thr)[0]
                for i in inds:
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = maskUtils.decode(segms[i]).astype(np.bool)
                    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
            # draw bounding boxes
            labels = [
                np.full(pts.shape[0], i, dtype=np.int32)
                for i, pts in enumerate(pts_result)
            ]
            labels = np.concatenate(labels)
            imshow_det_pts(
                img_show,
                bboxes,
                pts,
                labels,
                show=show,
                draw_pts=draw_pts,
                out_file=out_file,
                class_names=class_names,
                score_thr=score_thr)


def pts2result(pts, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, pts_num)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if pts.shape[0] == 0:
        return [
            np.zeros((0, pts.shape[1]), dtype=np.float32) for i in range(num_classes - 1)
        ]
    else:
        pts = pts.cpu().numpy()
        labels = labels.cpu().numpy()
        return [pts[labels == i, :] for i in range(num_classes - 1)]


def imshow_det_pts(img,
                   bboxes,
                   pts,
                   labels,
                   class_names=None,
                   score_thr=0,
                   thickness=1,
                   font_scale=0.5,
                   show=True,
                   win_name='',
                   wait_time=0,
                   draw_pts=True,
                   out_file=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert pts.ndim == 2
    assert labels.ndim == 1
    assert pts.shape[0] == labels.shape[0]
    img = imread(img)

    if score_thr > 0:
        scores = pts[:, -1]
        inds = scores > score_thr
        pts = pts[inds, :]
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    for pt, bbox, label in zip(pts, bboxes, labels):
        r_color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        pt_x = pt[0:-1:3]
        pt_y = pt[1:-1:3]
        pt_score = pt[2:-1:3]
        left_top = (int(bbox[0]), int(bbox[1]))
        right_bottom = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(
            img, left_top, right_bottom, r_color, thickness=thickness)
        if draw_pts:
            for i in range(pt_x.shape[0]):
                r_fore_color = r_color
                r_back_color = [int(i / 2) for i in r_color]
                r_show_color = r_fore_color if pt_score[i] > 0.5 else r_back_color
                cv2.circle(img, (pt_x[i], pt_y[i]), 1, r_show_color, thickness=2)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(pt) % 2 == 1:
            label_text += '|{:.02f}'.format(pt[-1])
        cv2.putText(img, label_text, (int(bbox[0]), int(bbox[1]) - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, r_color)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow(img, win_name='', wait_time=0):
    """
    Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)
