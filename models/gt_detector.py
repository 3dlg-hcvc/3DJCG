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
        max_num_proposal = 128
        data_dict["proposal_features"] = torch.zeros(batch_size, max_num_proposal, self.m, device="cuda")
        data_dict["proposal_centers"] = torch.zeros(batch_size, max_num_proposal, 3, device="cuda")
        data_dict["proposal_sizes"] = torch.zeros(batch_size, max_num_proposal, 3, device="cuda")


        for b in range(batch_size):

            proposal_batch_idx = torch.nonzero(data_dict["proposals_batchId"] == b).squeeze(-1)
            pred_num = len(proposal_batch_idx)
            data_dict["proposal_features"][b, :pred_num, :] = data_dict["proposal_feats"][proposal_batch_idx][:pred_num]
            data_dict["proposal_centers"][b, :pred_num, :] = data_dict["proposal_crop_bbox"][proposal_batch_idx, :3][:pred_num]
            data_dict["proposal_sizes"][b, :pred_num, :] = data_dict["proposal_crop_bbox"][proposal_batch_idx, 3:6][
                                                             :pred_num]
        del data_dict["proposal_crop_bbox"]
        return data_dict


    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        #### backbone
        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()]  # (N, m)

        num_proposals = len(data_dict["instances_bboxes_tmp"])# TODO  convert to batch
        gt_proposal_features = torch.empty(size=(num_proposals, pt_feats.shape[1]), device="cuda")

        batch_idxs = data_dict["locs_scaled"][:, 0].int()
        proposals_batchId_all = batch_idxs[data_dict["gt_proposals_idx"][:, 1].long()].int()

        proposals_batchId = proposals_batchId_all[data_dict["gt_proposals_offset"][:-1].long()]
        sem_labels = torch.empty(size=(num_proposals, ), device="cuda")

        for idx in range(num_proposals):
            start_idx = data_dict["gt_proposals_offset"][idx]
            end_idx = data_dict["gt_proposals_offset"][idx+1]
            proposal_info = data_dict["gt_proposals_idx"][start_idx:end_idx]
            proposal_point_mask = proposal_info[:, 1].long()
            proposal_features = torch.mean(pt_feats[proposal_point_mask], dim=0)
            gt_proposal_features[idx] = proposal_features
            sem_labels[idx] = 0

        data_dict["proposals_batchId"] = proposals_batchId
        data_dict["proposal_feats"] = gt_proposal_features
        data_dict["proposal_objectness_scores"] = torch.ones(size=(num_proposals,), dtype=torch.int32, device="cuda")

        proposal_crop_bbox = torch.zeros(num_proposals, 9, device="cuda")  # (nProposals, center+size+heading+label)
        proposal_crop_bbox[:, :3] = data_dict["instances_bboxes_tmp"][:, :3]
        proposal_crop_bbox[:, 3:6] = data_dict["instances_bboxes_tmp"][:, 3:6]
        proposal_crop_bbox[:, 7] = sem_labels
        proposal_crop_bbox[:, 8] = torch.ones(size=(num_proposals,), dtype=torch.int32, device="cuda")
        data_dict["proposal_crop_bbox"] = proposal_crop_bbox
        return data_dict


    def feed(self, data_dict):

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"], 4)  # (M, C), float, cuda
        data_dict = self.forward(data_dict)
        data_dict = self.convert_stack_to_batch(data_dict)

        return data_dict
