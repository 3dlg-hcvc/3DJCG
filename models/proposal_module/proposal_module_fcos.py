"""
Modified from: https://github.com/facebookresearch/votenet/blob/master/models/proposal_module.py
"""

import torch
import torch.nn as nn
import numpy as np
from lib.configs.config_joint import CONF


from lib.pointnet2.pointnet2_modules import PointnetSAModuleVotes
# from utils.box_util import get_3d_box_batch_of_rois_tensor, rotz_batch_pytorch
from utils.box_util import get_3d_box_batch, rotz_batch_pytorch
from models.proposal_module.ROI_heads.roi_heads import StandardROIHeads
from macro import *


class ProposalModule(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, seed_feat_dim=256, config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.num_proposal = num_proposal
        self.sampling = sampling
        self.seed_feat_dim = seed_feat_dim
        self.config = config
        # Vote clustering
        self.vote_aggregation = PointnetSAModuleVotes(
            npoint=self.num_proposal,
            radius=0.3,
            nsample=16,
            mlp=[self.seed_feat_dim, 128, 128, 128],
            use_xyz=True,
            normalize_xyz=True
        )
        # Object proposal/detection
        # Objectness scores (2), center residual (3),
        # heading class+residual (num_heading_bin*2), size class+residual(num_size_cluster*4)
        self.proposal = StandardROIHeads(num_heading_bin=num_heading_bin, num_class=num_class, seed_feat_dim=256)

    def forward(self, xyz, features, data_dict):
        """
        Args:
            xyz: (B,K,3)
            features: (B,C,K)
        Returns:
            scores: (B,num_proposal,2+3+NH*2+NS*4)
        """

        # Farthest point sampling (FPS) on votes
        if not USE_GT:
            xyz, features, fps_inds = self.vote_aggregation(xyz, features)

            sample_inds = fps_inds

            data_dict['aggregated_vote_xyz'] = xyz # (batch_size, num_proposal, 3)
            data_dict['aggregated_vote_features'] = features.permute(0, 2, 1).contiguous() # (batch_size, num_proposal, 128)
            data_dict['aggregated_vote_inds'] = sample_inds # (batch_size, num_proposal,) # should be 0,1,2,...,num_proposal
            # --------- PROPOSAL GENERATION ---------
            data_dict = self.proposal(features, data_dict, config=self.config)
            data_dict = self.decode_scores(data_dict)
        else:

            xyz = data_dict['center_label'][:, :, 0:3]
            data_dict['aggregated_vote_xyz'] = xyz
            data_dict['aggregated_vote_features'] = features
            features = features.permute(0, 2, 1)
            data_dict["pred_bbox_feature"] = data_dict["aggregated_vote_features"]

            # gt_bboxes_centers = data_dict["center_label"].cpu().numpy()
            # gt_heading_class_labels = data_dict["heading_class_label"].cpu().numpy()
            # gt_heading_residual_labels = data_dict["heading_residual_label"].cpu().numpy()
            # gt_size_class_labels = data_dict["size_class_label"].cpu().numpy()
            # gt_bboxes_residuals = data_dict["size_residual_label"].cpu().numpy()
            #
            # gt_obb_batch_new = self.config.param2obb_batch(gt_bboxes_centers, gt_heading_class_labels,
            #                                           gt_heading_residual_labels,
            #                                           gt_size_class_labels, gt_bboxes_residuals)
            # pred_bboxes = get_3d_box_batch(gt_obb_batch_new[:, 3:6], gt_obb_batch_new[:, 6],
            #                                      gt_obb_batch_new[:, 0:3])

            # batch_size, num_proposals, 8, 3

            data_dict['pred_heading'] = torch.zeros(size=(data_dict["heading_class_label"].shape[0], data_dict["heading_class_label"].shape[1]), dtype=torch.float32, device="cuda")
            data_dict['pred_center'] = data_dict["center_label"]
            data_dict["pred_size"] = data_dict["gt_size"]

            pred_heading = data_dict['pred_heading'].detach().cpu().numpy()  # B,num_proposal
            pred_center = data_dict['pred_center'].detach().cpu().numpy()  # (B, num_proposal)
            pred_box_size = data_dict['pred_size'].detach().cpu().numpy()  # (B, num_proposal, 3)

            pred_bboxes = get_3d_box_batch(pred_box_size, pred_heading, pred_center)
            pred_bboxes = torch.from_numpy(pred_bboxes).float().to(
                "cuda")  # .reshape(bsize, num_proposal, 8, 3)
            data_dict['pred_bbox_corner'] = pred_bboxes


        # net = self.proposal(features)
        # data_dict = self.decode_scores(net, data_dict, self.num_class, self.num_heading_bin, self.num_size_cluster, self.mean_size_arr)

        return data_dict

    def decode_pred_box(self, data_dict):
        # predicted bbox
        aggregated_vote_xyz = data_dict["aggregated_vote_xyz"] # (B,K,3)
        pred_heading_class = torch.argmax(data_dict["heading_scores"], -1) # B,num_proposal
        pred_heading_residual = torch.gather(data_dict["heading_residuals"], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
        rois = data_dict['rois'].contiguous()

        pred_heading = pred_heading_class.float() * (2.0*np.pi/self.num_heading_bin) + pred_heading_residual[..., 0]
        data_dict['pred_heading'] = pred_heading

        pred_box_size = rois[:, :, 0:3] + rois[:, :, 3:6]  # (B, N, 3)
        data_dict['pred_size'] = pred_box_size

        bsize = pred_box_size.shape[0]
        num_proposal = pred_box_size.shape[1]

        # Compute pred center xyz
        vote_xyz = (rois[:,:,0:3] - rois[:,:,3:6]) / 2  # (B, N, 3)
        R = rotz_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        # R = roty_batch_pytorch(pred_heading.float()).view(-1, 3, 3)
        vote_xyz = torch.matmul(vote_xyz.reshape(-1, 3).unsqueeze(1), R).squeeze(2)  # (B, N, 3)
        vote_xyz = vote_xyz.view(bsize, num_proposal, 3)
        pred_center = aggregated_vote_xyz - vote_xyz
        data_dict['pred_center'] = pred_center

        # batch_size, num_proposals, 8, 3
        pred_bboxes = get_3d_box_batch(pred_box_size.detach().cpu().numpy(),
                                       pred_heading.detach().cpu().numpy(), pred_center.detach().cpu().numpy())
        pred_bboxes = torch.from_numpy(pred_bboxes).float().to(pred_center.device)# .reshape(bsize, num_proposal, 8, 3)
        data_dict['pred_bbox_corner'] = pred_bboxes

        # Testing Scripts
        # center = pred_center[0, 13].cpu().numpy()
        # heading = pred_heading[0, 13].cpu().numpy()
        # size = pred_box_size[0, 13].cpu().numpy()
        # from utils.box_util import get_3d_box
        # newbox = get_3d_box(size, heading, center)

        # print(newbox, pred_bboxes[0, 13])
        # import ipdb
        # ipdb.set_trace()
        # # print(pred_center, pred_size)
        return data_dict

    def decode_scores(self, data_dict):
        """
        decode the predicted parameters for the bounding boxes

        """
        # processed box info
        data_dict = self.decode_pred_box(data_dict) # bounding box corner coordinates
        data_dict["pred_bbox_feature"] = data_dict["aggregated_vote_features"]
        # Not Useful
        data_dict["pred_bbox_mask"] = data_dict["objectness_scores"].argmax(-1)
        # data_dict["pred_bbox_sems"] = data_dict["sem_cls_scores"].argmax(-1)

        return data_dict

