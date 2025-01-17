# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np

from scipy.optimize import linear_sum_assignment
# sys.path.append(os.path.join(os.getcwd(), "lib")) # HACK add the lib folder
from .loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

from .loss_detection import compute_vote_loss, compute_objectness_loss, compute_box_loss, compute_box_and_sem_cls_loss

FAR_THRESHOLD = 0.3
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness

from macro import *


def compute_reference_loss(data_dict, config, no_reference=False):
    """ Compute cluster reference loss

    Args:
        data_dict: dict (read-only)

    Returns:
        ref_loss, lang_loss, cluster_preds, cluster_labels
    """


    # unpack
    # cluster_preds = data_dict["cluster_ref"] # (B, num_proposal)

    # predicted bbox
    pred_heading = data_dict['pred_heading'].detach().cpu().numpy() # B,num_proposal
    pred_center = data_dict['pred_center'].detach().cpu().numpy() # (B, num_proposal)
    pred_box_size = data_dict['pred_size'].detach().cpu().numpy() # (B, num_proposal, 3)


    box_mask = data_dict["ref_box_label_list"].to(torch.float32)
    gt_center_list = torch.einsum("abc,adb->adc", data_dict["center_label"], box_mask).cpu().numpy()
    gt_size_class_list = torch.einsum("ab,acb->ac", data_dict["size_class_label"].to(torch.float32), box_mask).to(
        torch.long).cpu().numpy()
    gt_size_residual_list = torch.einsum("abc,adb->adc", data_dict["size_residual_label"], box_mask).cpu().numpy()
    gt_heading_class_list = torch.einsum("ab,acb->ac", data_dict["heading_class_label"].to(torch.float32),
                                         box_mask).to(torch.long).cpu().numpy()
    gt_heading_residual_list = torch.einsum("ab,acb->ac", data_dict["heading_residual_label"].to(torch.float32),
                                            box_mask).to(torch.long).cpu().numpy()
    # gt_center_list = data_dict['ref_center_label_list'].cpu().numpy()  # (B,3)
    # gt_heading_class_list = data_dict['ref_heading_class_label_list'].cpu().numpy()  # B
    # gt_heading_residual_list = data_dict['ref_heading_residual_label_list'].cpu().numpy()  # B
    # gt_size_class_list = data_dict['ref_size_class_label_list'].cpu().numpy()  # B
    # gt_size_residual_list = data_dict['ref_size_residual_label_list'].cpu().numpy()  # B,3

    if SCANREFER_ENHANCE:
        box_mask_new = data_dict["multi_ref_box_label_list"]
        gt_box_num = data_dict["gt_box_num_list"].cpu().numpy()


    # convert gt bbox parameters to bbox corners
    batch_size, num_proposals = data_dict['pred_center'].shape[:2]
    batch_size, len_nun_max = data_dict["multi_ref_box_label_list"].shape[:2]
    lang_num = data_dict["lang_num"]
    max_iou_rate_25 = 0
    max_iou_rate_5 = 0

    if not no_reference:
        cluster_preds = data_dict["cluster_ref"].reshape(batch_size, len_nun_max, num_proposals)
    else:
        cluster_preds = torch.zeros((batch_size, len_nun_max, num_proposals), device="cuda")

    # print("cluster_preds",cluster_preds.shape)
    if not SCANREFER_ENHANCE:
        criterion = SoftmaxRankingLoss()
    else:
        criterion = nn.MultiLabelSoftMarginLoss()

    loss = 0.
    gt_labels = np.zeros((batch_size, len_nun_max, num_proposals))
    for i in range(batch_size):
        if not USE_GT:
            objectness_masks = data_dict['objectness_scores'].max(2)[
                1].float().cpu().numpy()  # batch_size, num_proposals
        else:
            objectness_masks = data_dict["tmp_objectness_masks"].squeeze().float().cpu().numpy()

        gt_obb_batch = config.param2obb_batch(gt_center_list[i][:, 0:3], gt_heading_class_list[i],
                                              gt_heading_residual_list[i],
                                              gt_size_class_list[i], gt_size_residual_list[i])
        gt_bbox_batch = get_3d_box_batch(gt_obb_batch[:, 3:6], gt_obb_batch[:, 6], gt_obb_batch[:, 0:3])

        labels = np.zeros((len_nun_max, num_proposals))

        labels_new = np.zeros((len_nun_max, num_proposals))

        pred_center_batch = pred_center[i]
        pred_heading_batch = pred_heading[i]
        pred_box_size_batch = pred_box_size[i]
        pred_bbox_batch = get_3d_box_batch(pred_box_size_batch, pred_heading_batch, pred_center_batch)

        for j in range(len_nun_max):
            if j < lang_num[i]:
                # convert the bbox parameters to bbox corners


                ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox_batch[j], (num_proposals, 1, 1)))

                if data_dict["istrain"][0] == 1 and not no_reference and data_dict["random"] < 0.5:
                    ious = ious * objectness_masks[i]

                ious_ind = ious.argmax()
                max_ious = ious[ious_ind]
                if max_ious >= 0.25:
                    labels[j, ious.argmax()] = 1  # treat the bbox with highest iou score as the gt
                    max_iou_rate_25 += 1
                if max_ious >= 0.5:
                    max_iou_rate_5 += 1


                if SCANREFER_ENHANCE:
                    single_box_mask = box_mask_new[i][j]
                    if single_box_mask.sum() == 0:
                        continue
                    gt_bboxes_centers = data_dict["center_label"][i][single_box_mask].cpu().numpy()
                    gt_heading_class_labels = data_dict["heading_class_label"][i][single_box_mask].cpu().numpy()
                    gt_heading_residual_labels = data_dict["heading_residual_label"][i][single_box_mask].cpu().numpy()
                    gt_size_class_labels = data_dict["size_class_label"][i][single_box_mask].cpu().numpy()
                    gt_bboxes_residuals = data_dict["size_residual_label"][i][single_box_mask].cpu().numpy()

                    gt_obb_batch_new = config.param2obb_batch(gt_bboxes_centers[:, 0:3], gt_heading_class_labels,
                                                          gt_heading_residual_labels,
                                                          gt_size_class_labels, gt_bboxes_residuals)
                    gt_bbox_batch_new = get_3d_box_batch(gt_obb_batch_new[:, 3:6], gt_obb_batch_new[:, 6], gt_obb_batch_new[:, 0:3])
                    iou_matrix = np.zeros(shape=(gt_bbox_batch_new.shape[0], ious.shape[0]))
                    for k, gt_bbox in enumerate(gt_bbox_batch_new):
                        ious = box3d_iou_batch(pred_bbox_batch, np.tile(gt_bbox, (num_proposals, 1, 1)))
                        if data_dict["istrain"][0] == 1 and not no_reference and data_dict["random"] < 0.5:
                            ious = ious * objectness_masks[i]
                        if SCANREFER_ENHANCE_VANILLE:
                            filtered_ious_indices = np.where(ious >= SCANREFER_ENHANCE_LOSS_THRESHOLD)
                            if filtered_ious_indices[0].shape[0] == 0:
                                continue
                            labels_new[j, filtered_ious_indices] = 1
                        else:
                            iou_matrix[k] = ious * -1
                    if not SCANREFER_ENHANCE_VANILLE:
                        row_idx, col_idx = linear_sum_assignment(iou_matrix)
                        for index in range(len(row_idx)):
                            if (iou_matrix[row_idx[index], col_idx[index]] * -1) >= SCANREFER_ENHANCE_LOSS_THRESHOLD:
                                labels_new[j, col_idx[index]] = 1


        cluster_labels = torch.cuda.FloatTensor(labels_new)  # B proposals
        gt_labels[i] = labels_new
        # reference loss
        loss += criterion(cluster_preds[i, :lang_num[i]], cluster_labels[:lang_num[i]].float().clone())

    data_dict['max_iou_rate_0.25'] = max_iou_rate_25 / sum(lang_num.cpu().numpy())
    data_dict['max_iou_rate_0.5'] = max_iou_rate_5 / sum(lang_num.cpu().numpy())

    # print("max_iou_rate", data_dict['max_iou_rate_0.25'], data_dict['max_iou_rate_0.5'])
    cluster_labels = torch.cuda.FloatTensor(gt_labels)  # B len_nun_max proposals
    # print("cluster_labels", cluster_labels.shape)
    loss = loss / batch_size
    # print("ref_loss", loss)
    return data_dict, loss, cluster_preds, cluster_labels


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    object_cat_list = data_dict["object_cat_list"]
    batch_size, len_nun_max = object_cat_list.shape[:2]
    lang_num = data_dict["lang_num"]
    lang_scores = data_dict["lang_scores"].reshape(batch_size, len_nun_max, -1)
    loss = 0.
    for i in range(batch_size):
        num = lang_num[i]
        loss += criterion(lang_scores[i, :num], object_cat_list[i, :num])
    loss = loss / batch_size
    return loss


# def get_loss(data_dict, config, detection=True, reference=True, use_lang_classifier=True):
#     """ Loss functions
#
#     Args:
#         data_dict: dict
#         config: dataset config instance
#         reference: flag (False/True)
#     Returns:
#         loss: pytorch scalar tensor
#         data_dict: dict
#     """
#     # Vote loss
#     if not USE_GT:
#         vote_loss = compute_vote_loss(data_dict)
#
#         # Obj loss
#         objectness_loss, objectness_label, objectness_mask, object_assignment = compute_objectness_loss(data_dict)
#         num_proposal = objectness_label.shape[1]
#         total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
#         data_dict["objectness_label"] = objectness_label
#         data_dict["objectness_mask"] = objectness_mask
#         data_dict["object_assignment"] = object_assignment
#         data_dict["pos_ratio"] = torch.sum(objectness_label.float())/float(total_num_proposal)
#         data_dict["neg_ratio"] = torch.sum(objectness_mask.float())/float(total_num_proposal) - data_dict["pos_ratio"]
#
#         # Box loss and sem cls loss
#         heading_cls_loss, heading_reg_loss, size_distance_loss, sem_cls_loss = compute_box_and_sem_cls_loss(data_dict, config)
#         box_loss = 0.1 * heading_cls_loss + heading_reg_loss + 0.1 * sem_cls_loss
#         box_loss = box_loss + 20 * size_distance_loss
#
#         # objectness; Nothing
#         obj_pred_val = torch.argmax(data_dict["objectness_scores"], 2) # B,K
#         obj_acc = torch.sum((obj_pred_val==data_dict["objectness_label"].long()).float()*data_dict["objectness_mask"])/(torch.sum(data_dict["objectness_mask"])+1e-6)
#         data_dict["obj_acc"] = obj_acc
#
#     if detection:
#         if not USE_GT:
#             data_dict["vote_loss"] = vote_loss
#             data_dict["objectness_loss"] = objectness_loss
#             data_dict["heading_cls_loss"] = heading_cls_loss
#             data_dict["heading_reg_loss"] = heading_reg_loss
#             data_dict["size_distance_loss"] = size_distance_loss
#             data_dict["sem_cls_loss"] = sem_cls_loss
#             data_dict["box_loss"] = box_loss
#     else:
#         if not USE_GT:
#             device = vote_loss.device()
#             data_dict["vote_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["objectness_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["heading_cls_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["heading_reg_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["size_distance_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["sem_cls_loss"] = torch.zeros(1)[0].to(device)
#             data_dict["box_loss"] = torch.zeros(1)[0].to(device)
#
#     if reference:
#         # Reference loss
#         data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config)
#         data_dict["cluster_labels"] = cluster_labels
#         data_dict["ref_loss"] = ref_loss
#     else:
#         #raise NotImplementedError('Only detection; not implemented')
#         # # Reference loss
#         data_dict, ref_loss, _, cluster_labels = compute_reference_loss(data_dict, config, no_reference=True)
#         lang_count = data_dict['ref_center_label_list'].shape[1]
#         # data_dict["cluster_labels"] = objectness_label.new_zeros(objectness_label.shape).cuda().repeat(lang_count, 1)
#         data_dict["cluster_labels"] = cluster_labels
#         data_dict["cluster_ref"] = objectness_label.new_zeros(objectness_label.shape, device="cuda", dtype=torch.float32).repeat(lang_count, 1)
#         # store
#         data_dict["ref_loss"] = torch.zeros(1)[0].cuda()
#         # data_dict['max_iou_rate_0.25'] = 0
#         # data_dict['max_iou_rate_0.5'] = 0
#
#     if reference and use_lang_classifier:
#         data_dict["lang_loss"] = compute_lang_classification_loss(data_dict)
#     else:
#         data_dict["lang_loss"] = torch.zeros(1)[0].cuda()
#
#     # Final loss function
#     # loss = data_dict['vote_loss'] + 0.1 * data_dict['objectness_loss'] + data_dict['box_loss'] + 0.1 * data_dict['sem_cls_loss'] + 0.03 * data_dict["ref_loss"] + 0.03 * data_dict["lang_loss"]
#     loss = 0
#
#     # Final loss function
#     if detection and not USE_GT:
#         # sem_cls loss is included in the box_loss
#         # detection_loss = detection_loss + 0.1 * data_dict['sem_cls_loss']
#         detection_loss = data_dict["vote_loss"] + 0.1*data_dict["objectness_loss"] + 1.0*data_dict["box_loss"]
#         detection_loss *= 10 # amplify
#         loss = loss + detection_loss
#     if reference:
#         loss = loss + 0.3 * data_dict["ref_loss"]
#     if use_lang_classifier:
#         loss = loss + 0.3 * data_dict["lang_loss"]
#     data_dict["loss"] = loss
#
#     return data_dict
