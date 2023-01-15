import torch
import functools
import torch.nn as nn
import MinkowskiEngine as ME
from lib.pointgroup_ops.functions import pointgroup_ops
from models.common import ResidualBlock, UBlock


class GTDetector(nn.Module):
    def __init__(self):
        super().__init__()

        in_channel = 3 + 3 + 128
        m = 16
        self.m = m
        D = 3
        blocks = [1, 2, 3, 4, 5, 6, 7]

        self.mode = 4

        block = ResidualBlock

        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)

        #### backbone
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m * c for c in blocks], sp_norm, 2, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

    @staticmethod
    def get_batch_offsets(batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1).int().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets

    def convert_stack_to_batch(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        max_num_proposal = 256
        data_dict["pred_bbox_feature"] = torch.zeros(size=(batch_size, max_num_proposal, self.m), device="cuda",
                                                 dtype=torch.float32)
        data_dict["objectness_scores"] = torch.zeros(size=(batch_size, max_num_proposal, 1), device="cuda", dtype=bool)
        data_dict["obj_features"] = torch.zeros(size=(batch_size, max_num_proposal, 128), device="cuda", dtype=torch.float32)

        # proposal_bbox = data_dict["proposal_crop_bbox"].detach().cpu().numpy()
        # proposal_bbox = get_3d_box_batch(proposal_bbox[:, :3], proposal_bbox[:, 3:6],
        #                                  proposal_bbox[:, 6])  # (nProposals, 8, 3)
        # proposal_bbox_tensor = torch.tensor(proposal_bbox).type_as(data_dict["proposal_feats"])
        #
        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(data_dict["proposals_batchId"] == b).squeeze(-1)
            pred_num = len(proposal_batch_idx)
            data_dict["pred_bbox_feature"][b, :pred_num, :] = data_dict["proposal_feats"][proposal_batch_idx][:pred_num]
            data_dict["objectness_scores"][b, :pred_num, 0] = data_dict["proposal_objectness_scores"][
                                                                  proposal_batch_idx][:pred_num]
            data_dict["obj_features"][b, :pred_num, :] = data_dict["tmp_obj_features"][proposal_batch_idx][:pred_num]

        data_dict["pred_center"] = data_dict['center_label']
        data_dict['pred_heading'] = data_dict['heading_class_label']
        data_dict['heading_residuals'] = data_dict['heading_residual_label']
        data_dict['size_scores'] = data_dict['size_class_label']
        data_dict['size_residuals'] = data_dict['size_residual_label']
        data_dict['objectness_label'] = data_dict["objectness_scores"]
        return data_dict


    def forward(self, data_dict):
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        #### backbone
        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()]  # (N, m)

        num_proposals = len(data_dict["gt_proposals_offset"]) - 1  # TODO  convert to batch
        gt_proposal_features = torch.empty(size=(num_proposals, pt_feats.shape[1]), device="cuda")

        gt_obj_features = torch.empty(size=(num_proposals, 128), device="cuda")

        batch_idxs = data_dict["locs_scaled"][:, 0].int()
        proposals_batchId_all = batch_idxs[data_dict["gt_proposals_idx"][:, 1].long()].int()

        proposals_batchId = proposals_batchId_all[data_dict["gt_proposals_offset"][:-1].long()]
        sem_labels = torch.empty(size=(num_proposals,), device="cuda")

        for idx in range(num_proposals):
            start_idx = data_dict["gt_proposals_offset"][idx]
            end_idx = data_dict["gt_proposals_offset"][idx + 1]
            proposal_info = data_dict["gt_proposals_idx"][start_idx:end_idx]
            proposal_point_mask = proposal_info[:, 1].long()

            gt_proposal_features[idx] = torch.mean(pt_feats[proposal_point_mask], dim=0)
            gt_obj_features[idx] = torch.mean(data_dict["feats"][:, 3:131][proposal_point_mask], dim=0)
            sem_labels[idx] = 0

        data_dict["proposals_batchId"] = proposals_batchId
        data_dict["proposal_feats"] = gt_proposal_features
        data_dict["proposal_objectness_scores"] = torch.ones(size=(num_proposals,), dtype=bool, device="cuda")
        data_dict["tmp_obj_features"] = gt_obj_features
        return data_dict


    def feed(self, data_dict):

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"], 4)  # (M, C), float, cuda
        data_dict = self.forward(data_dict)
        data_dict = self.convert_stack_to_batch(data_dict)

        return data_dict
