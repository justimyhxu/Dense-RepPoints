from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import (PointGenerator, multiclass_nms_pts,
                        dense_reppoints_target, multi_apply)
from mmdet.ops import ConvModule
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob


@HEADS.register_module
class DenseRepPointsHead(nn.Module):
    """
    Dense RepPoints head
    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        stacked_mask_convs(int): How man conv layers for mask pts branch.
        num_groups(int): The number of groups for group pooling for classification.
        use_sparse_pts_for_cls(bool): Whether use sparse group pooling for classification.
        num_points(int): Points numbef of Dense RepPoints Set.
        num_score_group(int): The group number of score map.
        gradient_mul (float): The multiplier to gradients from points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial bbox loss.
        loss_bbox_refine (dict): Config of bbox loss in refinement.
        loss_bbox_border_init(dict): Config of initial points border loss.
        loss_bbox_border_refine(dict): Config of the border loss in refinement.
        loss_pts_init(dict): Config of the initial points distribution loss.
        loss_pts_refine(dict): Config of points distribution loss in refinement.
        loss_mask_score_init(dict): Config of initial points score loss.
        transform_method(str): The methods to transform RepPoints to bbox.
        sample_padding_mode(str): Padding mode for sampling features.
        fuse_mask_feat(Bool): Whether to fuse mask feature.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 stacked_mask_convs=3,
                 num_group=9,
                 use_sparse_pts_for_cls=False,
                 num_points=9,
                 num_score_group=1,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
                 loss_bbox_init=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_bbox_border_init=dict(type='PtsBorderLoss', loss_weight=0.25),
                 loss_bbox_border_refine=dict(type='PtsBorderLoss', loss_weight=0.5),
                 loss_pts_init=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=0.5, eps=1e-12),
                 loss_pts_refine=dict(type='ChamferLoss2D', use_cuda=True, loss_weight=1.0, eps=1e-12),
                 loss_mask_score_init=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                 transform_method='minmax',
                 sample_padding_mode='border',
                 fuse_mask_feat=False,
                 **kwargs
                 ):
        super(DenseRepPointsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.stacked_mask_convs = stacked_mask_convs
        self.num_points = num_points
        self.group_num = num_group
        self.divisible_group = self.num_points % self.group_num == 0
        self.use_sparse_pts_for_cls = use_sparse_pts_for_cls
        self.gradient_mul = gradient_mul

        self.num_score_group = num_score_group
        score_group_kernel = int(np.sqrt(self.num_score_group))
        assert score_group_kernel ** 2 == self.num_score_group, 'Group number must be a square number'
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.num_lvls = len(self.point_strides)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_bbox_border_init = build_loss(loss_bbox_border_init)
        self.loss_bbox_border_refine = build_loss(loss_bbox_border_refine)
        self.loss_pts_init = build_loss(loss_pts_init)
        self.loss_pts_refine = build_loss(loss_pts_refine)
        self.loss_mask_score_init = build_loss(loss_mask_score_init)

        self.fuse_mask_feat = fuse_mask_feat
        self.sample_padding_mode = sample_padding_mode
        self.transform_method = transform_method

        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = []
        for _ in self.point_strides:
            self.point_generators.append(PointGenerator())
        self._init_dcn_offset(num_points)
        self._init_layers()

    def _init_dcn_offset(self, num_points):
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, "The points number should be an odd square number."
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.mask_convs = nn.ModuleList()
        self.reppoints_cls = nn.ModuleList()
        self.reppoints_pts_init = nn.ModuleList()
        self.reppoints_pts_refine = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        for i in range(self.stacked_mask_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.mask_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))

        pts_out_dim = 2 * self.num_points
        self.reppoints_cls_conv = nn.Conv2d(self.feat_channels * self.group_num, self.point_feat_channels, 1, 1, 0)
        self.reppoints_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)
        self.reppoints_pts_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_pts_refine_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

        pts_out_dim = 2 * self.num_points
        self.reppoints_pts_refine_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)

        self.reppoints_mask_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.reppoints_mask_init_out = nn.Conv2d(self.point_feat_channels, self.num_score_group, 1, 1, 0)

        if self.fuse_mask_feat:
            self.mask_fuse_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.mask_convs:
            normal_init(m.conv, std=0.01)
        if self.fuse_mask_feat:
            normal_init(self.mask_fuse_conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.reppoints_cls_conv, std=0.01)
        normal_init(self.reppoints_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_pts_init_conv, std=0.01)
        normal_init(self.reppoints_pts_init_out, std=0.01)
        normal_init(self.reppoints_pts_refine_conv, std=0.01)
        normal_init(self.reppoints_pts_refine_out, std=0.01)
        normal_init(self.reppoints_mask_init_conv, std=0.01)
        normal_init(self.reppoints_mask_init_out, std=0.01)

    def transform_box(self, pts, y_first=True):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [x1, y1, x2, y2] transformed from points.
        """
        if self.transform_method == 'minmax':
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
            pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        else:
            raise NotImplementedError
        return bbox

    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
        regressed bboxes and generate the grids on the bboxes.

            Args:
                reg(Tensor): the regression value to previous bboxes.
                previous_boxes(Tensor): previous bboxes.
            Returns:
                Tensor: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        tx = reg[:, [0], ...]
        ty = reg[:, [1], ...]
        tw = reg[:, [2], ...]
        th = reg[:, [3], ...]
        bx = (previous_boxes[:, [0], ...] + previous_boxes[:, [2], ...]) / 2.
        by = (previous_boxes[:, [1], ...] + previous_boxes[:, [3], ...]) / 2.
        bw = (previous_boxes[:, [2], ...] - previous_boxes[:, [0], ...]).clamp(min=1e-6)
        bh = (previous_boxes[:, [3], ...] - previous_boxes[:, [1], ...]).clamp(min=1e-6)
        grid_left = bx + bw * tx - 0.5 * bw * torch.exp(tw)
        grid_width = bw * torch.exp(tw)
        grid_up = by + bh * ty - 0.5 * bh * torch.exp(th)
        grid_height = bh * torch.exp(th)
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_up + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([grid_left, grid_up, grid_left + grid_width, grid_up + grid_height], 1)
        return grid_yx, regressed_bbox

    def sample_offset(self, x, flow, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor): input feature, size (n, c, h, w)
                flow (Tensor): flow fields, size(n, 2, h', w')
                padding_mode (str): grid sample padding mode, 'zeros' or 'border'
            Returns:
                Tensor: warped feature map (n, c, h', w')
        """
        # assert x.size()[-2:] == flow.size()[-2:]
        n, _, h, w = flow.size()
        x_ = torch.arange(w).view(1, -1).expand(h, -1)
        y_ = torch.arange(h).view(-1, 1).expand(-1, w)
        grid = torch.stack([x_, y_], dim=0).float().cuda()
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1)
        grid = grid + flow
        gx = 2 * grid[:, 0, :, :] / (w - 1) - 1
        gy = 2 * grid[:, 1, :, :] / (h - 1) - 1
        grid = torch.stack([gx, gy], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        if torch.__version__ >= '1.3.0':
            return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)
        else:
            return F.grid_sample(x, grid, padding_mode=padding_mode)

    def compute_offset_feature(self, x, offset, padding_mode):
        """
        sample feature based on offset

            Args:
                x (Tensor) : feature map, size (n, C, h, w),  x first
                offset (Tensor) : offset, size (n, sample_pts*2, h, w), x first
                padding_mode (str): 'zeros' or 'border' or 'relection'
            Returns:
                Tensor: the warped feature generated by the offset and the input feature map, size (n, sample_pts, C, h, w)
        """
        offset_reshape = offset.view(offset.shape[0], -1, 2, offset.shape[2], offset.shape[
            3])  # (n, sample_pts, 2, h, w)
        num_pts = offset_reshape.shape[1]
        offset_reshape = offset_reshape.contiguous().view(-1, 2, offset.shape[2],
                                                          offset.shape[3])  # (n*sample_pts, 2, h, w)
        x_repeat = x.unsqueeze(1).repeat(1, num_pts, 1, 1, 1)  # (n, sample_pts, C, h, w)
        x_repeat = x_repeat.view(-1, x_repeat.shape[2], x_repeat.shape[3], x_repeat.shape[4])  # (n*sample_pts, C, h, w)
        sampled_feat = self.sample_offset(x_repeat, offset_reshape, padding_mode)  # (n*sample_pts, C, h, w)
        sampled_feat = sampled_feat.view(-1, num_pts, sampled_feat.shape[1], sampled_feat.shape[2],
                                         sampled_feat.shape[3])  # (n, sample_pts, C, h, w)
        return sampled_feat

    def sample_offset_3d(self, x, flow, padding_mode):
        """
        sample feature based on 2D offset(x, y) + 1-D index(z)

            Args:
                x (Tensor): size (n, c, d', h', w')
                flow (Tensor): size(n, 3, d, h, w)
                padding_mode (str): 'zeros' or 'border'
            Returns:
                warped feature map generated by the offset and the input feature map, size(n, c, d, h, w)
        """
        n, _, d, h, w = flow.size()
        group_num = x.shape[2]
        device = flow.get_device()
        x_ = torch.arange(w, device=device).view(1, 1, -1).expand(d, h, -1).float()  # (d, h, w)
        y_ = torch.arange(h, device=device).view(1, -1, 1).expand(d, -1, w).float()  # (d, h, w)
        z_ = torch.zeros(d, h, w, device=device)  # (d, h, w)
        grid = torch.stack([x_, y_, z_], dim=0).float()  # (3, d, h, w)
        del x_, y_, z_
        grid = grid.unsqueeze(0).expand(n, -1, -1, -1, -1)  # (n, 3, d, h, w)
        grid = grid + flow  # (n, 3, d, h, w)
        gx = 2 * grid[:, 0, :, :, :] / (w - 1) - 1  # (n, d, h, w)
        gy = 2 * grid[:, 1, :, :, :] / (h - 1) - 1  # (n, d, h, w)
        gz = 2 * grid[:, 2, :, :, :] / (group_num - 1) - 1  # (n, d, h, w)
        grid = torch.stack([gx, gy, gz], dim=1)  # (n, 3, d, h, w)
        del gx, gy, gz
        grid = grid.permute(0, 2, 3, 4, 1)  # (n, d, h, w, 3)
        if torch.__version__ >= '1.3.0':
            return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)
        else:
            return F.grid_sample(x, grid, padding_mode=padding_mode)

    def compute_offset_feature_5d(self, x, offset, padding_mode):
        """
        sample 5D feature based on offset

            Args:
                x (Tensor) : input feature, size (n, C, d', h', w'), x first
                offset (Tensor) : flow field, size (n, 3, sample_pts, h, w), x first
                padding_mode (str): 'zeros' or 'border'
            Returns:
                Tensor: offset_feature, size (n, sample_pts, C, h, w)
        """
        sampled_feat = self.sample_offset_3d(x, offset, padding_mode)  # (n, C, sample_pts, h, w)
        sampled_feat = sampled_feat.transpose(1, 2)  # (n, sample_pts, C, h, w)
        return sampled_feat

    def aggregate_offset_feature(self, x, offset, padding_mode):
        """
        sample feature based on offset and setlect feature by max pooling

            Args:
                x (Tensor) : input feature, size (b, C, h, w), x first
                offset (Tensor) : flow field, size (n, 2, sample_pts, h, w), x first
                padding_mode (str): 'zeros' or 'border'
            Returns:
                Tensor: offset_feature, size (n, C, h, w)
        """
        feature = self.compute_offset_feature(x, offset, padding_mode=padding_mode)  # (b, n, C, h, w)
        feature = feature.max(dim=1)[0]  # (b, C, h, w)
        return feature

    def forward_pts_head_single(self, x, cur_lvl=None):
        b, _, h, w = x.shape
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        scale = self.point_base_scale / 2
        points_init = dcn_base_offset / dcn_base_offset.max() * scale

        cls_feat = x
        pts_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        # generate points_init
        pts_out_init = self.reppoints_pts_init_out(self.relu(self.reppoints_pts_init_conv(pts_feat)))
        pts_out_init = pts_out_init + points_init  # (b, 2n, h, w)
        pts_out_init_detach = (1 - self.gradient_mul) * pts_out_init.detach() + self.gradient_mul * pts_out_init
        # classify dense reppoints based on group pooling
        if not self.use_sparse_pts_for_cls:
            if self.divisible_group:
                group_card = int(self.num_points * 2 / self.group_num)
                cls_offset = pts_out_init_detach.view(b, self.group_num, group_card, h, w)  # (b, g, n/g, h, w)
                cls_pts_feature = [
                    self.aggregate_offset_feature(cls_feat, cls_offset[:, i, ...],
                                                  padding_mode=self.sample_padding_mode)
                    for i in range(self.group_num)]  # list of (b, c, h, w)
                cls_pts_feature = torch.stack(cls_pts_feature, dim=1)  # (b, g, C, h, w)
                cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)
                cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))
            else:
                assert self.group_num > 1
                group_card = int(np.floor(self.num_points / self.group_num) * 2)
                main_idxs = torch.arange(0, group_card * (self.group_num - 1), dtype=torch.long)
                res_idxs = torch.arange(group_card * (self.group_num - 1), 2 * self.num_points, dtype=torch.long)
                cls_offset_main = pts_out_init_detach[:, main_idxs, :, :] \
                    .view(b, self.group_num - 1, group_card, h, w)  # (b, g-1, n/g, h, w)
                cls_offset_res = pts_out_init_detach[:, res_idxs, :, :].view(b, 1, -1, h, w)  # (b, 1, *, h, w)
                cls_pts_feature = [
                    self.aggregate_offset_feature(cls_feat, cls_offset_main[:, i, ...],
                                                  padding_mode=self.sample_padding_mode)
                    for i in range(self.group_num - 1)]  # list of (b, c, h, w)
                cls_pts_feature.append(self.aggregate_offset_feature(cls_feat, cls_offset_res[:, 0, ...],
                                                                     padding_mode=self.sample_padding_mode))
                cls_pts_feature = torch.stack(cls_pts_feature, dim=1)  # (b, g, C, h, w)
                cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)
                cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))
        else:
            if self.divisible_group:
                cls_offset = pts_out_init_detach.view(b, self.group_num, -1, 2, h, w)
                cls_offset = cls_offset[:, :, 0, ...].reshape(b, -1, h, w)
            else:
                assert self.group_num > 1
                pts_idxs = torch.from_numpy(np.floor(np.linspace(0, self.num_points, self.group_num, endpoint=False))
                                            ).long().to(pts_out_init_detach.device)
                cls_offset = pts_out_init_detach.view(b, -1, 2, h, w)[:, pts_idxs, ...]
                cls_offset = cls_offset.reshape(b, -1, h, w)
            cls_pts_feature = self.compute_offset_feature(cls_feat, cls_offset, padding_mode=self.sample_padding_mode)
            cls_pts_feature = cls_pts_feature.contiguous().view(b, -1, h, w)
            cls_out = self.reppoints_cls_out(self.relu(self.reppoints_cls_conv(cls_pts_feature)))
        # generate offset field
        pts_refine_field = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat)))  # (b, n*2, h, w)
        pts_refine_field = pts_refine_field.view(b * self.num_points, -1, h, w)  # (b*n, 2, h, w)
        pts_out_init_detach_reshape = pts_out_init_detach.view(b, -1, 2, h, w).view(-1, 2, h, w)  # (b*n, 2, h, w)
        pts_out_refine = self.compute_offset_feature(
            pts_refine_field, pts_out_init_detach_reshape,
            padding_mode=self.sample_padding_mode)  # (b*n, 2, h, w)
        pts_out_refine = pts_out_refine.view(b, -1, h, w)  # (b, n*2, h, w)
        # generate points_refine
        pts_out_refine = pts_out_refine + pts_out_init_detach

        return cls_out, pts_out_init, pts_out_refine

    @staticmethod
    def normalize_pts_within_bboxes(pts):
        """
        Normalize pts offset within bboxes(instance level)

            Args:
                pts(Tensor): input points, size (b, n, 2, h_pts, w_pts)

            Returns:
                Tensor: normalized_pts, size (b, n, 2, h_pts, w_pts)
        """
        b, _, _, h_pts, w_pts = pts.shape
        _pts_x = pts[:, :, 0, :, :]  # (b, n, h_pts, w_pts)
        _pts_y = pts[:, :, 1, :, :]  # (b, n, h_pts, w_pts)
        _bbox_left = torch.min(_pts_x, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_right = torch.max(_pts_x, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_bottom = torch.max(_pts_y, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_up = torch.min(_pts_y, dim=1, keepdim=True)[0]  # (b, 1, h_pts, w_pts)
        _bbox_w = _bbox_right - _bbox_left  # (b, 1, h_pts, w_pts)
        _bbox_h = _bbox_bottom - _bbox_up  # (b, 1, h_pts, w_pts)

        normalized_x = (_pts_x - _bbox_left) / (_bbox_w + 1e-6)  # (b, n, h_pts, w_pts)
        normalized_y = (_pts_y - _bbox_up) / (_bbox_h + 1e-6)  # (b, n, h_pts, w_pts)
        normalized_pts = torch.stack([normalized_x, normalized_y], dim=2)  # (b, n, 2, h_pts, w_pts)
        return normalized_pts

    def grid_position_sensitive_group_partition(self, pts, num_group):
        """
        Position-sensitive group partition based on grids.

            Args:
                pts(Tensor): input points, size (b, n, 2, h_pts, w_pts)
                num_group(int): the number of groups

            Returs:
                Tensor: group_inds, size (b, 1, n, h_pts, w_pts)
        """
        normalized_pts = self.normalize_pts_within_bboxes(pts)  # (b, n, 2, h_pts, w_pts)
        normalized_x = normalized_pts[:, :, 0, :, :]  # (b, n, h_pts, w_pts)
        normalized_y = normalized_pts[:, :, 1, :, :]  # (b, n, h_pts, w_pts)

        num_group_kernel = int(np.sqrt(num_group))
        grid_x_inds = (normalized_x * num_group_kernel).long()  # (b, n, h_pts, w_pts)
        grid_y_inds = (normalized_y * num_group_kernel).long()  # (b, n, h_pts, w_pts)
        group_inds = grid_y_inds * num_group_kernel + grid_x_inds  # (b, n, h_pts, w_pts)
        group_inds = group_inds.unsqueeze(1).float()  # (b, 1, n, h_pts, w_pts)
        return group_inds

    def forward_mask_head_single(self, pts, mask_feat):
        b, _, h, w = mask_feat.shape
        h_pts, w_pts = pts.shape[-2:]
        score_map = self.reppoints_mask_init_out(
            self.relu(self.reppoints_mask_init_conv(mask_feat)))  # (b, G*1, h, w)
        # position sensitive group partition based on grids
        pts_reshape_detach = pts.detach().view(b, -1, 2, h_pts, w_pts)  # (b, n, 2, h_pts, w_pts)
        group_inds = self.grid_position_sensitive_group_partition(
            pts_reshape_detach, self.num_score_group)  # (b, 1, n, h_pts, w_pts)
        del pts_reshape_detach
        score_map = score_map.unsqueeze(1)  # (b, 1, G, h, w)

        pts_reshape = pts.view(b, -1, 2, h_pts, w_pts).transpose(1, 2)  # (b, 2, n, h_pts, w_pts)
        pts_reshape = pts_reshape.detach()
        _pts_inds_cat = torch.cat([pts_reshape, group_inds], dim=1)  # (b, 3, n, h_pts, w_pts)
        del group_inds, pts_reshape
        # position sensitive sampling on score maps
        pts_score_out = self.compute_offset_feature_5d(
            score_map, _pts_inds_cat, padding_mode=self.sample_padding_mode)  # (b, n, 1, h_pts, w_pts)

        pts_score_out = pts_score_out.view(b, -1, h_pts, w_pts)  # (b, n, h_pts, w_pts)
        return pts_score_out, _

    def forward_mask_head(self, mask_feat_list, pts_out_list):
        for mask_conv in self.mask_convs:
            mask_feat_list = [mask_conv(mask_feat) for mask_feat in mask_feat_list]
        if self.fuse_mask_feat:
            mask_feat_high_res = mask_feat_list[0]
            H, W = mask_feat_high_res.shape[-2:]
            mask_feat_up_list = []
            for lvl, mask_feat in enumerate(mask_feat_list):
                lambda_val = 2.0 ** lvl
                mask_feat_up = mask_feat
                if lvl > 0:
                    mask_feat_up = F.interpolate(
                        mask_feat, scale_factor=lambda_val, mode="bilinear", align_corners=False
                    )
                    del mask_feat
                mask_feat_up_list.append(
                    self.mask_fuse_conv(mask_feat_up[:, :, :H, :W] + mask_feat_high_res)
                )
                del mask_feat_up
            del mask_feat_high_res
            del mask_feat_list
            mask_feat_list = mask_feat_up_list

        pts_score_out = multi_apply(self.forward_mask_head_single, pts_out_list, mask_feat_list)[0]
        return pts_score_out

    def forward(self, feats_list, test=False):
        cls_out_list, pts_out_init_list, pts_out_refine_list = multi_apply(
            self.forward_pts_head_single, feats_list, range(self.num_lvls))
        if test:
            pts_out_list = pts_out_refine_list
        else:
            pts_out_list = [(1 - self.gradient_mul) * pts_out_init.detach()
                            + self.gradient_mul * pts_out_init for pts_out_init in pts_out_init_list]

        pts_score_out = self.forward_mask_head(feats_list, pts_out_list)
        return cls_out_list, pts_out_init_list, pts_out_refine_list, pts_score_out

    def get_points(self, featmap_sizes, img_metas):
        """
        Get points according to feature map sizes.

            Args:
                featmap_sizes (list[tuple]): Multi-level feature map sizes.
                img_metas (list[dict]): Image meta info.

            Returns:
                tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags((feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def centers_to_bboxes(self, point_list):
        """
        Get bboxes according to center points. Only used in MaxIOUAssigner.

            Args:
                point_list(list(Tensor): Multi image points list for different level

            Returns:
                list(Tensor): the bbox converted from center points and organized with the same order as input point_list
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale, scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def bboxes_to_centers(self, bbox_list):
        """
        Get center points according to bboxes. Only used in MaxIOUAssigner.

            Args:
                bbox_list(list(Tensor)): Multi image bbox list for differnet level

            Returns:
                list(Tensor): center points converted from bbox and organized withe the same order as input bbox
        """
        point_list = []
        for i_lvl in range(len(self.point_strides)):
            point_x = (bbox_list[i_lvl][..., 0] + bbox_list[i_lvl][..., 2]) / 2.
            point_y = (bbox_list[i_lvl][..., 1] + bbox_list[i_lvl][..., 3]) / 2.
            point_stride = (bbox_list[i_lvl][..., 2] - bbox_list[i_lvl][..., 0]) / self.point_base_scale
            point = torch.stack([point_x, point_y, point_stride], 2)
            point_list.append(point)
        return point_list

    def pts_to_img(self, center_list, pred_list, y_first=True):
        """
        Project points offset based on center point to image scale and stack different level projected points

            Args:
                center_list(list(Tensor)): Multi image center list with different level
                pred_list: Multi image pred points offset with different level
                y_first(bool): f y_fisrt=True, the point set is represented as
                    [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                    represented as [x1, y1, x2, y2 ... xn, yn].
            Returns:
                list(Tensor): multi-image points in image scale and stack points in different level
        """
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                if y_first:
                    yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                    pts_y = yx_pts_shift[..., 0::2]
                    pts_x = yx_pts_shift[..., 1::2]
                    xy_pts_shift = torch.stack([pts_x, pts_y], -1)
                    xy_pts_shift = xy_pts_shift.view(*pts.shape[:-1], -1)
                else:
                    xy_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    # pts_to_img_lvl
    def pts_to_img_lvl(self, center_list, pred_list, y_first=True):
        """
        Project points offset based on center point to image scale and organized in image-level order

            Args:
                center_list(list(Tensor)): Multi image center list with different level
                pred_list: Multi image pred points offset with different level
                y_first(bool): f y_fisrt=True, the point set is represented as
                    [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                    represented as [x1, y1, x2, y2 ... xn, yn].
            Returns:
                list(Tensor): multi-image points in image scale with different level
        """
        pts_list = []
        for i_img, point in enumerate(center_list):
            pts_img = []
            for i_lvl in range(len(center_list[0])):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                if y_first:
                    yx_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                    pts_y = yx_pts_shift[..., 0::2]
                    pts_x = yx_pts_shift[..., 1::2]
                    xy_pts_shift = torch.stack([pts_x, pts_y], -1)
                    xy_pts_shift = xy_pts_shift.view(*pts.shape[:-1], -1)
                else:
                    xy_pts_shift = pts_shift.permute(1, 2, 0).view(-1, 2 * self.num_points)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_img.append(pts)
            pts_list.append(pts_img)
        return pts_list

    def loss_single(self, cls_score, pts_pred_init, pts_pred_refine, pts_score_pred_init,
                    labels, label_weights,
                    bbox_gt_init, pts_gt_init, bbox_weights_init,
                    bbox_gt_refine, pts_gt_refine, pts_score_gt_label, bbox_weights_refine,
                    stride, num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples_refine)

        # bbox loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.transform_box(pts_pred_init.reshape(-1, 2 * self.num_points), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.transform_box(pts_pred_refine.reshape(-1, 2 * self.num_points), y_first=False)
        normalize_term = self.point_base_scale * stride
        loss_bbox_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_bbox_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)

        # pts border loss
        loss_border_init = self.loss_bbox_border_init(
            pts_pred_init.reshape(-1, 2 * self.num_points) / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            y_first=False,
            avg_factor=num_total_samples_init
        ) if self.loss_bbox_border_init is not None else loss_bbox_init.new_zeros(1)
        loss_border_refine = self.loss_bbox_border_refine(
            pts_pred_refine.reshape(-1, 2 * self.num_points) / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            y_first=False,
            avg_factor=num_total_samples_refine
        ) if self.loss_bbox_border_refine is not None else loss_bbox_refine.new_zeros(1)

        # pts_loss_init
        valid_pts_gt_init = torch.cat(pts_gt_init, 0)
        valid_pts_gt_init = valid_pts_gt_init.view(-1, self.num_points, 2)
        mask_pred_init = pts_pred_init.reshape(-1, 2 * self.num_points)
        valid_pts_pred_init = mask_pred_init[bbox_weights_init[:, 0] > 0]
        valid_pts_pred_init = valid_pts_pred_init.view(-1, self.num_points, 2)
        valid_pts = valid_pts_gt_init.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_init, 1)
        loss_pts_init = self.loss_pts_init(
            valid_pts_gt_init[valid_pts] / normalize_term,
            valid_pts_pred_init[valid_pts] / normalize_term).sum() / num_total_samples
        # pts_loss_refine
        valid_pts_gt_refine = torch.cat(pts_gt_refine, 0)
        valid_pts_gt_refine = valid_pts_gt_refine.view(-1, self.num_points, 2)
        pts_pred_refine = pts_pred_refine.reshape(-1, 2 * self.num_points)
        valid_pts_pred_refine = pts_pred_refine[bbox_weights_refine[:, 0] > 0]
        valid_pts_pred_refine = valid_pts_pred_refine.view(-1, self.num_points, 2)
        valid_pts = valid_pts_gt_refine.sum(-1).sum(-1) > 0
        num_total_samples = max(num_total_samples_refine, 1)
        loss_pts_refine = self.loss_pts_refine(
            valid_pts_gt_refine[valid_pts] / normalize_term,
            valid_pts_pred_refine[valid_pts] / normalize_term).sum() / num_total_samples
        # mask score loss
        valid_pts_score_gt_label = torch.cat(pts_score_gt_label, 0)
        valid_pts_score_gt_label = valid_pts_score_gt_label.view(-1, self.num_points, 1)
        pts_score_pred_init = pts_score_pred_init.reshape(-1, self.num_points)
        valid_pts_score_pred_init = pts_score_pred_init[bbox_weights_refine[:, 0] > 0]
        valid_pts_score_pred_init = valid_pts_score_pred_init.view(-1, self.num_points, 1)
        valid_pts_score_inds = (valid_pts_score_gt_label.sum(-1).sum(-1) > 0)
        num_total_samples = max(num_total_samples_refine, 1)
        loss_mask_score_init = self.loss_mask_score_init(
            valid_pts_score_pred_init[valid_pts_score_inds],
            valid_pts_score_gt_label[valid_pts_score_inds],
            weight=bbox_weights_init.new_ones(*valid_pts_score_pred_init[valid_pts_score_inds].shape),
            avg_factor=num_total_samples
        ) / self.num_points

        return loss_cls, loss_bbox_init, loss_pts_init, loss_bbox_refine, loss_pts_refine, loss_mask_score_init, \
               loss_border_init, loss_border_refine

    def loss(self,
             cls_scores,
             pts_preds_init,
             pts_preds_refine,
             pts_preds_score_init,
             gt_bboxes,
             gt_masks,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_init = self.pts_to_img(proposal_list, pts_preds_init, y_first=False)
        proposal_pts_list = self.pts_to_img_lvl(proposal_list, pts_preds_init, y_first=False)
        real_pts_preds_score_init = []
        for lvl_pts_score in pts_preds_score_init:
            b = lvl_pts_score.shape[0]
            real_pts_preds_score_init.append(lvl_pts_score.permute(0, 2, 3, 1).view(b, -1, self.num_points))
        if cfg.init.assigner['type'] != 'PointBoxAssigner':
            proposal_list = self.centers_to_bboxes(proposal_list)

        cls_reg_targets_init = dense_reppoints_target(
            proposal_list,
            proposal_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            num_pts=self.num_points)

        (*_, bbox_gt_list_init, pts_gt_list_init, _, proposal_list_init,
         bbox_weights_list_init, num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (num_total_pos_init + num_total_neg_init if self.sampling else num_total_pos_init)

        # target for refinement stage
        proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        real_pts_preds_refine = self.pts_to_img(proposal_list, pts_preds_refine, y_first=False)
        bbox_pts_list = self.pts_to_img_lvl(proposal_list, pts_preds_init, y_first=False)

        bbox_list = []
        for i_img, point in enumerate(proposal_list):
            bbox = []
            for i_lvl in range(len(pts_preds_refine)):
                bbox_preds_init = self.transform_box(pts_preds_init[i_lvl].detach(), y_first=False)
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).contiguous().view(-1, 4))
            bbox_list.append(bbox)
        cls_reg_targets_refine = dense_reppoints_target(
            bbox_list,
            bbox_pts_list,
            valid_flag_list,
            gt_bboxes,
            gt_masks,
            img_metas,
            cfg.refine,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling,
            num_pts=self.num_points)
        (labels_list, label_weights_list, bbox_gt_list_refine, pts_gt_list_refine, pts_score_gt_label_list,
         proposal_list_refine,
         bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine
        num_total_samples_refine = (
            num_total_pos_refine + num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        losses_cls, losses_bbox_init, losses_pts_init, losses_bbox_refine, losses_pts_refine, losses_mask_score_init, \
        losses_border_init, losses_border_refine = multi_apply(
            self.loss_single,
            cls_scores,
            real_pts_preds_init,
            real_pts_preds_refine,
            real_pts_preds_score_init,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            pts_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            pts_gt_list_refine,
            pts_score_gt_label_list,
            bbox_weights_list_refine,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_refine)
        loss_dict_all = {'loss_cls': losses_cls,
                         'loss_bbox_init': losses_bbox_init,
                         'losses_pts_init': losses_pts_init,
                         'losses_bbox_refine': losses_bbox_refine,
                         'losses_pts_refines': losses_pts_refine,
                         'losses_mask_score_init': losses_mask_score_init,
                         'losses_border_init': losses_border_init,
                         'losses_border_refine': losses_border_refine
                         }
        return loss_dict_all

    def get_bboxes(self, cls_scores, pts_preds_init, pts_preds_refine, pts_preds_score_refine, img_metas, cfg,
                   rescale=False, nms=True):
        assert len(cls_scores) == len(pts_preds_refine)
        bbox_preds_refine = [self.transform_box(pts_pred_refine, y_first=False) for pts_pred_refine in pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            pts_pred_list = [
                pts_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            mask_pred_list = [
                pts_preds_score_refine[i][img_id].sigmoid().detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list, pts_pred_list, mask_pred_list,
                                               mlvl_points, img_shape, scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          pts_preds,
                          mask_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_pts = []
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_masks = []
        for i_lvl, (cls_score, bbox_pred, pts_pred, mask_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, pts_preds, mask_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            pts_pred = pts_pred.permute(1, 2, 0).reshape(-1, 2 * self.num_points)
            mask_pred = mask_pred.permute(1, 2, 0).reshape(-1, self.num_points)

            # mask scoring
            mask_sum = (mask_pred > 0.5).sum(1).float()
            mask_score = ((mask_pred > 0.5).float() * mask_pred).sum(1) / (mask_sum + 1e-6)
            scores = scores * mask_score.unsqueeze(1)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                pts_pred = pts_pred[topk_inds, :]
                mask_pred = mask_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[i_lvl] + pts_pos_center
            pts[:, 0::2] = pts[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            pts[:, 1::2] = pts[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            bboxes[:, 0::2] = bboxes[:, 0::2].clamp(min=0, max=img_shape[1] - 1)
            bboxes[:, 1::2] = bboxes[:, 1::2].clamp(min=0, max=img_shape[0] - 1)

            mlvl_pts.append(pts)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_masks.append(mask_pred)
        mlvl_pts = torch.cat(mlvl_pts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_pts /= mlvl_pts.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_masks = torch.cat(mlvl_masks)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            # moddify
            det_bboxes, det_pts, det_masks, det_labels = multiclass_nms_pts(
                mlvl_bboxes, mlvl_pts, mlvl_scores, mlvl_masks, cfg.score_thr, cfg.nms, cfg.max_per_img)
            return det_bboxes, det_pts, det_masks, det_labels
        else:
            return mlvl_bboxes, mlvl_pts, mlvl_masks, mlvl_scores
