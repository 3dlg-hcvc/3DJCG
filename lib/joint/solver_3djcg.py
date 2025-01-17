'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
#import shutil
from lib.configs.config_joint import CONF
from lib.loss_helper.loss_joint import get_joint_loss
#from lib.joint.eval_caption import eval_cap
#from lib.joint.eval_ground import get_eval as eval_ground
#from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler
#from macro import *
from lib.multi3drefer_evaluator import Multi3DReferEvaluator
import wandb
from utils.box_util import get_3d_box_batch

# ITER_REPORT_TEMPLATE = """
# -------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
# [loss] train_loss: {train_loss}
# [loss] train_ref_loss: {train_ref_loss}
# [loss] train_lang_loss: {train_lang_loss}
# [loss] train_cap_loss: {train_cap_loss}
# [loss] train_ori_loss: {train_ori_loss}
# [loss] train_dist_loss: {train_dist_loss}
# [loss] train_objectness_loss: {train_objectness_loss}
# [loss] train_vote_loss: {train_vote_loss}
# [loss] train_box_loss: {train_box_loss}
# [loss] train_lang_acc: {train_lang_acc}
# [sco.] train_ref_acc: {train_ref_acc}
# [sco.] train_cap_acc: {train_cap_acc}
# [sco.] train_ori_acc: {train_ori_acc}
# [sco.] train_obj_acc: {train_obj_acc}
# [sco.] train_pred_ious: {train_pred_ious}
# [sco.] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
# [sco.] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
# [sco.] train_iou_max_rate_0.25: {train_iou_max_rate_25}, train_iou_max_rate_0.5: {train_iou_max_rate_5}
# [info] mean_fetch_time: {mean_fetch_time}s
# [info] mean_forward_time: {mean_forward_time}s
# [info] mean_backward_time: {mean_backward_time}s
# [info] mean_eval_time: {mean_eval_time}s
# [info] mean_iter_time: {mean_iter_time}s
# [info] mean_real_time: {mean_real_time}s
# [info] ETA: {eta_h}h {eta_m}m {eta_s}s
# """
#
# EPOCH_REPORT_TEMPLATE = """
# ---------------------------------summary---------------------------------
# [train] train_bleu-1: {train_bleu_1}
# [train] train_bleu-2: {train_bleu_2}
# [train] train_bleu-3: {train_bleu_3}
# [train] train_bleu-4: {train_bleu_4}
# [train] train_cider: {train_cider}
# [train] train_rouge: {train_rouge}
# [train] train_meteor: {train_meteor}
# [train] train_loss: {train_loss}
# [train] train_ref_loss: {train_ref_loss}
# [train] train_lang_loss: {train_lang_loss}
# [train] train_objectness_loss: {train_objectness_loss}
# [train] train_vote_loss: {train_vote_loss}
# [train] train_box_loss: {train_box_loss}
# [train] train_lang_acc: {train_lang_acc}
# [train] train_ref_acc: {train_ref_acc}
# [train] train_obj_acc: {train_obj_acc}
# [train] train_pos_ratio: {train_pos_ratio}, train_neg_ratio: {train_neg_ratio}
# [train] train_iou_rate_0.25: {train_iou_rate_25}, train_iou_rate_0.5: {train_iou_rate_5}
# [train] train_max_iou_rate_0.25: {train_max_iou_rate_25}, train_max_iou_rate_0.5: {train_max_iou_rate_5}
# [val]   val_bleu-1: {val_bleu_1}
# [val]   val_bleu-2: {val_bleu_2}
# [val]   val_bleu-3: {val_bleu_3}
# [val]   val_bleu-4: {val_bleu_4}
# [val]   val_cider: {val_cider}
# [val]   val_rouge: {val_rouge}
# [val]   val_meteor: {val_meteor}
# [val]   val_loss: {val_loss}
# [val]   val_ref_loss: {val_ref_loss}
# [val]   val_lang_loss: {val_lang_loss}
# [val]   val_objectness_loss: {val_objectness_loss}
# [val]   val_vote_loss: {val_vote_loss}
# [val]   val_box_loss: {val_box_loss}
# [val]   val_lang_acc: {val_lang_acc}
# [val]   val_ref_acc: {val_ref_acc}
# [val]   val_obj_acc: {val_obj_acc}
# [val]   val_pos_ratio: {val_pos_ratio}, val_neg_ratio: {val_neg_ratio}
# [val]   val_iou_rate_0.25: {val_iou_rate_25}, val_iou_rate_0.5: {val_iou_rate_5}
# [val]   val_max_iou_rate_0.25: {val_max_iou_rate_25}, val_max_iou_rate_0.5: {val_max_iou_rate_5}
# [val]   scanrefer_plus_overall_25: {scanrefer_plus_overall_25}, scanrefer_plus_overall_50: {scanrefer_plus_overall_50},
# """
#
# BEST_REPORT_TEMPLATE = """
# --------------------------------------best--------------------------------------
# [best] epoch: {epoch}
# [best]  bleu-1: {bleu_1}
# [best]  bleu-2: {bleu_2}
# [best]  bleu-3: {bleu_3}
# [best]  bleu-4: {bleu_4}
# [best]  cider: {cider}
# [best]  rouge: {rouge}
# [best]  meteor: {meteor}
# [sco.] ref_acc: {ref_acc}
# [sco.] obj_acc: {obj_acc}
# [sco.] pos_ratio: {pos_ratio}, neg_ratio: {neg_ratio}
# [sco.] iou_rate_0.25: {iou_rate_25}, iou_rate_0.5: {iou_rate_5}
# [best] best_caption_epoch: {best_caption_epoch}
# [best]  best_caption_bleu-4: {best_caption_bleu_4}
# [best]  best_caption_cider: {best_caption_cider}
# [best]  best_caption_rouge: {best_caption_rouge}
# [best]  best_caption_meteor: {best_caption_meteor}
# [sco.] best_ground_epoch: {best_ground_epoch}
# [sco.] best_ground_iou_rate_0.25: {best_ground_iou_rate_25}, best_ground_iou_rate_0.5: {best_ground_iou_rate_5}
# """

def get_eval_multi3drefer(data_dict, config):
    batch_size, lang_chunk_size = data_dict["ann_id"].shape

    pred_aabb_score_masks = (
            torch.sigmoid(data_dict["cluster_ref"]) >= 0.1
    ).reshape(shape=(batch_size, lang_chunk_size, -1)).cpu().numpy()

    pred_results = {}

    # pred_center = data_dict['center'].detach().cpu().numpy()  # (B,K,3)
    # pred_heading_class = torch.argmax(data_dict['heading_scores'], -1)  # B,num_proposal
    # pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2,
    #                                      pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    # pred_heading_class = pred_heading_class.detach().cpu().numpy()  # B,num_proposal
    # pred_heading_residual = pred_heading_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal
    # pred_size_class = torch.argmax(data_dict['size_scores'], -1)  # B,num_proposal
    # pred_size_residual = torch.gather(data_dict['size_residuals'], 2,
    #                                   pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
    #                                                                                      3))  # B,num_proposal,1,3
    # pred_size_class = pred_size_class.detach().cpu().numpy()
    # pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy()  # B,num_proposal,3
    gts = {}
    new_box_mask = data_dict["multi_ref_box_label_list"]

    # box_mask = data_dict["ref_box_label_list"].to(torch.float32)

    pred_heading = data_dict['pred_heading'].detach().cpu().numpy()  # B,num_proposal
    pred_center = data_dict['pred_center'].detach().cpu().numpy()  # (B, num_proposal)
    pred_box_size = data_dict['pred_size'].detach().cpu().numpy()  # (B, num_proposal, 3)

    for i in range(batch_size):

        pred_center_batch = pred_center[i]
        pred_heading_batch = pred_heading[i]
        pred_box_size_batch = pred_box_size[i]
        pred_bbox_batch = get_3d_box_batch(pred_box_size_batch, pred_heading_batch, pred_center_batch)

        # pred_obb_batch = config.param2obb_batch(pred_center[i, :, 0:3], pred_heading_class[i],
        #                                         pred_heading_residual[i],
        #                                         pred_size_class[i], pred_size_residual[i])
        # pred_bbox_corners = get_3d_box(pred_obb_batch[:, 3:6], pred_obb_batch[:, 6], pred_obb_batch[:, 0:3])
        min_max_bound = np.stack((pred_bbox_batch.min(1), pred_bbox_batch.max(1)), axis=1)

        for j in range(lang_chunk_size):


            if j < data_dict["lang_num"][i]:
                pred_aabbs = min_max_bound[pred_aabb_score_masks[i, j]]
                pred_results[
                    (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                     data_dict["ann_id"][i][j].item())
                ] = {
                    "aabb_bound": pred_aabbs
                }

            single_box_mask = new_box_mask[i][j]

            gt_bboxes_centers = data_dict["center_label"][i][single_box_mask].cpu().numpy()
            gt_heading_class_labels = data_dict["heading_class_label"][i][single_box_mask].cpu().numpy()
            gt_heading_residual_labels = data_dict["heading_residual_label"][i][single_box_mask].cpu().numpy()
            gt_size_class_labels = data_dict["size_class_label"][i][single_box_mask].cpu().numpy()
            gt_bboxes_residuals = data_dict["size_residual_label"][i][single_box_mask].cpu().numpy()

            gt_obb_batch_new = config.param2obb_batch(gt_bboxes_centers[:, 0:3], gt_heading_class_labels,
                                                      gt_heading_residual_labels,
                                                      gt_size_class_labels, gt_bboxes_residuals)
            gt_bbox_batch_new = get_3d_box_batch(gt_obb_batch_new[:, 3:6], gt_obb_batch_new[:, 6],
                                                 gt_obb_batch_new[:, 0:3])

            # gt_bboxes = gt_bboxes_list[i][single_mask]
            gt_bboxes_bound = np.stack((gt_bbox_batch_new.min(1), gt_bbox_batch_new.max(1)), axis=1)

            gts[
                (data_dict["scene_id"][i], data_dict["object_id"][i][j].item(),
                 data_dict["ann_id"][i][j].item())
            ] = {
                "aabb_bound": gt_bboxes_bound,
                "eval_type": data_dict["eval_type"][j][i]
            }


    # batch_size = len(data_dict["scene_id"])
    # chunk_size = data_dict["ann_id"].shape[1]
    # box_masks = data_dict["multi_ref_box_label_list"]

    # gt_bboxes_list = data_dict["gt_bbox"]



    # gt_target_obj_id_masks = data_dict["gt_target_obj_id_mask"].permute(1, 0)FF
    # for i in range(batch_size):
    #     # aabb_start_idx = data_dict["aabb_count_offsets"][i]
    #     # aabb_end_idx = data_dict["aabb_count_offsets"][i + 1]
    #
    #
    #     for j in range(chunk_size):
    #
    #         if j < data_dict["lang_num"][i]:

    return pred_results, gts

class Solver():
    def __init__(self, model, device, config, dataset, dataloader, optimizer, stamp, val_step=10, num_ground_epoch=50,
    detection=True, reference=True, use_lang_classifier=True, caption=True, orientation=False, distance=False, use_tf=True,
    lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None,
    criterion="meteor", checkpoint_best=None):

        self.epoch = 0                    # set in __call__
        self.verbose = 0                  # set in __call__
        
        self.model = model
        self.device = device
        self.config = config
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step
        self.num_ground_epoch = num_ground_epoch

        self.detection = detection
        self.reference = reference
        self.use_lang_classifier = use_lang_classifier
        self.caption = caption
        self.orientation = orientation
        self.distance = distance
        self.use_tf = use_tf

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        # self.criterion = criterion
        # self.checkpoint_best = checkpoint_best

        self.evaluator = Multi3DReferEvaluator(verbose=False, metric_name="f1")

        run = wandb.init(
            project="3djcg", name=self.stamp
        )
        # self.best = {
        #     "epoch": 0,
        #     "best_caption_epoch": 0,
        #     "best_ground_epoch": 0,
        #     "bleu-1": -float("inf"),
        #     "bleu-2": -float("inf"),
        #     "bleu-3": -float("inf"),
        #     "bleu-4": -float("inf"),
        #     "cider": -float("inf"),
        #     "rouge": -float("inf"),
        #     "meteor": -float("inf"),
        #     "best_caption_bleu-4": -float("inf"),
        #     "best_caption_cider": -float("inf"),
        #     "best_caption_rouge": -float("inf"),
        #     "best_caption_meteor": -float("inf"),
        #     "sum": -float("inf"),
        #     "ground_sum": -float("inf"),
        #     "caption_sum": -float("inf"),
        #     "lang_acc": -float("inf"),
        #     "ref_acc": -float("inf"),
        #     "obj_acc": -float("inf"),
        #     "pos_ratio": -float("inf"),
        #     "neg_ratio": -float("inf"),
        #     "iou_rate_0.25": -float("inf"),
        #     "iou_rate_0.5": -float("inf"),
        #     "best_ground_iou_rate_0.25": -float("inf"),
        #     "best_ground_iou_rate_0.5": -float("inf"),
        #     "max_iou_rate_0.25": -float("inf"),
        #     "max_iou_rate_0.5": -float("inf"),
        #     "scanrefer++_overall_25": -float("inf"),
        #     "scanrefer++_overall_50": -float("inf"),
        # } if checkpoint_best == None else checkpoint_best

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }
        
        # tensorboard
        # os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        # os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        # self._log_writer = {
        #     "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
        #     "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        # }

        # # training log
        # log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        # self.log_fout = open(log_path, "a")
        #
        # eval_path = os.path.join(CONF.PATH.OUTPUT, stamp, "eval.txt")
        # self.eval_fout = open(eval_path, "a")
        #
        # # private
        # # only for internal access and temporary results
        # self._running_log = {}
        # self._global_iter_id = 0
        self._total_iter = {}             # set in __call__

        # # templates
        # self.__iter_report_template = ITER_REPORT_TEMPLATE
        # self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        # self.__best_report_template = BEST_REPORT_TEMPLATE

        if lr_decay_step:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            elif isinstance(lr_decay_step, dict):
                if lr_decay_step['type'] != 'cosine':
                    raise NotImplementedError('lr dict type should be cosine (other not implemented)')
                print(lr_decay_step, '<< lr_decay_step dict', flush=True)
                config = lr_decay_step
                config['optimizer'] = optimizer
                config.pop('type')
                self.lr_scheduler = CosineAnnealingLR(**config)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate**(int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = (len(self.dataloader["eval"]["train"]) + len(self.dataloader["eval"]["val"])) \
             * (self._total_iter["train"] / self.val_step)
        
        for epoch_id in range(epoch):
            print(f"Current Epoch=> {epoch_id}")
            torch.cuda.empty_cache()

            # self._log("epoch {} starting...".format(epoch_id + 1))

            if self.lr_scheduler:
                # self.lr_scheduler.step()
                print("learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()), flush=True)
                # now_lr = self.lr_scheduler.get_lr()[0]
                for (idx, param_group) in enumerate(self.optimizer.param_groups):
                    # print(param_group.keys(), '<< param key shape')
                    print('[LR Param Group]', param_group['Param_Name'], param_group['lr'], '<< should', flush=True)
                    # param_group['lr'] = base_group_lr[idx] / base_lr * now_lr

            # feed
            self.dataloader['train'].dataset.shuffle_data()
            self._feed(self.dataloader["train"], "train", epoch_id)


            # update lr scheduler
            if self.lr_scheduler:
                print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                self.lr_scheduler.step()

            # update bn scheduler
            if self.bn_scheduler:
                print("update batch normalization momentum --> {}\n".format(self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                self.bn_scheduler.step()

            if epoch_id % 10 == 0:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    self._feed(self.dataloader["eval"]["val"], "val", epoch_id, is_eval=True)
                    # self._dump_log("val", True)
                    # self._epoch_report(epoch_id)
                    # self._finish(epoch_id)
                


        # finish training
        # self._finish(epoch_id)
    #
    # def _log(self, info_str):
    #     self.log_fout.write(info_str + "\n")
    #     self.log_fout.flush()
    #     print(info_str, flush=True)
    #
    # def _log_eval(self, info_str):
    #     self.eval_fout.write(info_str + "\n")
    #     self.eval_fout.flush()
    #     print(info_str, flush=True)
    #
    # def _reset_log(self, phase):
    #     if phase == "train":
    #         self.log[phase] = {
    #             # info
    #             "forward": [],
    #             "backward": [],
    #             "eval": [],
    #             "fetch": [],
    #             "iter_time": [],
    #             "real_time": [],
    #             # loss (float, not torch.cuda.FloatTensor)
    #             "loss": [],
    #             "ref_loss": [],
    #             "lang_loss": [],
    #             "cap_loss": [],
    #             "ori_loss": [],
    #             "dist_loss": [],
    #             "objectness_loss": [],
    #             "vote_loss": [],
    #             "box_loss": [],
    #             # scores (float, not torch.cuda.FloatTensor)
    #             "lang_acc": [],
    #             "ref_acc": [],
    #             "cap_acc": [],
    #             "ori_acc": [],
    #             "obj_acc": [],
    #             "pred_ious": [],
    #             "pos_ratio": [],
    #             "neg_ratio": [],
    #             "iou_rate_0.25": [],
    #             "iou_rate_0.5": [],
    #             "max_iou_rate_0.25": [],
    #             "max_iou_rate_0.5": [],
    #             # for eval
    #             "bleu-1": [],
    #             "bleu-2": [],
    #             "bleu-3": [],
    #             "bleu-4": [],
    #             "cider": [],
    #             "rouge": [],
    #             "meteor": []
    #         }
    #     else:
    #         self.log[phase] = {
    #             "bleu-1": [],
    #             "bleu-2": [],
    #             "bleu-3": [],
    #             "bleu-4": [],
    #             "cider": [],
    #             "rouge": [],
    #             "meteor": [],
    #             # info
    #             "forward": [],
    #             "backward": [],
    #             "eval": [],
    #             "fetch": [],
    #             "iter_time": [],
    #             "real_time": [],
    #             # loss (float, not torch.cuda.FloatTensor)
    #             "loss": [],
    #             "ref_loss": [],
    #             "lang_loss": [],
    #             "cap_loss": [],
    #             "ori_loss": [],
    #             "dist_loss": [],
    #             "objectness_loss": [],
    #             "vote_loss": [],
    #             "box_loss": [],
    #             # scores (float, not torch.cuda.FloatTensor)
    #             "lang_acc": [],
    #             "ref_acc": [],
    #             "cap_acc": [],
    #             "ori_acc": [],
    #             "obj_acc": [],
    #             "pred_ious": [],
    #             "pos_ratio": [],
    #             "neg_ratio": [],
    #             "iou_rate_0.25": [],
    #             "iou_rate_0.5": [],
    #             "max_iou_rate_0.25": [],
    #             "max_iou_rate_0.5": []
    #         }

    # def _dump_log(self, phase, is_eval=False):
    #     if phase == "train" and not is_eval:
    #         log = {
    #             "loss": ["loss", "ref_loss", "lang_loss", "cap_loss", "ori_loss", "dist_loss", "objectness_loss", "vote_loss", "box_loss"],
    #             "score": ["lang_acc", "ref_acc", "cap_acc", "ori_acc", "obj_acc", "pred_ious", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5", "max_iou_rate_0.25", "max_iou_rate_0.5"]
    #         }
    #         for key in log:
    #             for item in log[key]:
    #                 if self.log[phase][item]:
    #                     self._log_writer[phase].add_scalar(
    #                         "{}/{}".format(key, item),
    #                         np.mean([v for v in self.log[phase][item]]),
    #                         self._global_iter_id
    #                     )
    #
    #     # eval
    #     if is_eval:
    #         log = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
    #         for key in log:
    #             if self.log[phase][key]:
    #                 self._log_writer[phase].add_scalar(
    #                     "eval/{}".format(key),
    #                     self.log[phase][key],
    #                     self._global_iter_id
    #                 )
    #         ground_log = {
    #             "score": ["lang_acc", "ref_acc", "obj_acc", "pos_ratio", "neg_ratio", "iou_rate_0.25", "iou_rate_0.5",
    #                       "max_iou_rate_0.25", "max_iou_rate_0.5"]
    #         }
    #         for key in ground_log:
    #             for item in ground_log[key]:
    #                 self._log_writer[phase].add_scalar(
    #                     "{}/{}".format(key, item),
    #                     np.mean([v for v in self.log[phase][item]]),
    #                     self._global_iter_id
    #                 )

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict, use_tf=self.use_tf)

        return data_dict

    def _backward(self, loss):
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_joint_loss(
            data_dict=data_dict, 
            device=self.device,
            config=self.config, 
            weights=self.dataset["train"].weights,
            detection=self.detection,
            reference=self.reference,
            use_lang_classifier=self.use_lang_classifier,
            caption=self.caption,
            orientation=self.orientation,
            distance=self.distance,
            num_ground_epoch=self.num_ground_epoch
        )

        # # store loss
        # self._running_log["ref_loss"] = data_dict["ref_loss"]
        # self._running_log["lang_loss"] = data_dict["lang_loss"]
        #
        # self._running_log["cap_loss"] = data_dict["cap_loss"]
        # self._running_log["ori_loss"] = data_dict["ori_loss"]
        # self._running_log["dist_loss"] = data_dict["dist_loss"]
        # if not USE_GT:
        #     self._running_log["objectness_loss"] = data_dict["objectness_loss"]
        #     self._running_log["vote_loss"] = data_dict["vote_loss"]
        #     self._running_log["box_loss"] = data_dict["box_loss"]
        #     self._running_log["obj_acc"] = data_dict["obj_acc"].item()
        #     self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
        #     self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
        # else:
        #     self._running_log["objectness_loss"] = 0
        #     self._running_log["vote_loss"] = 0
        #     self._running_log["box_loss"] = 0
        #     self._running_log["obj_acc"] = 0
        #     self._running_log["pos_ratio"] = 0
        #     self._running_log["neg_ratio"] = 0
        #
        # self._running_log["loss"] = data_dict["loss"]
        #
        # # store eval
        # self._running_log["cap_acc"] = data_dict["cap_acc"].item()
        # self._running_log["ori_acc"] = data_dict["ori_acc"].item()
        # self._running_log["pred_ious"] = data_dict["pred_ious"].item()
        # self._running_log["max_iou_rate_0.25"] = np.mean(data_dict["max_iou_rate_0.25"])
        # self._running_log["max_iou_rate_0.5"] = np.mean(data_dict["max_iou_rate_0.5"])



    # def _ground_eval(self, data_dict, phase, is_eval, mem_hash=None, final_output=None):
    #     if phase == "train" and is_eval == False:
    #         data_dict = eval_ground(
    #             data_dict=data_dict,
    #             config=self.config,
    #             reference=self.reference,
    #             use_lang_classifier=self.use_lang_classifier,
    #         )
    #         # dump
    #         # self._running_log["lang_acc"] = data_dict["lang_acc"].item()
    #         # self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
    #         # if not USE_GT:
    #         #     self._running_log["objectness_loss"] = data_dict["objectness_loss"]
    #         #     self._running_log["vote_loss"] = data_dict["vote_loss"]
    #         #     self._running_log["box_loss"] = data_dict["box_loss"]
    #         #     self._running_log["obj_acc"] = data_dict["obj_acc"].item()
    #         #     self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
    #         #     self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
    #         # self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
    #         # self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
    #         # self._running_log["max_iou_rate_0.25"] = np.mean(data_dict["max_iou_rate_0.25"])
    #         # self._running_log["max_iou_rate_0.5"] = np.mean(data_dict["max_iou_rate_0.5"])
    #
    #     # elif phase == "val" and is_eval == True:
    #     #     data_dict = eval_ground(
    #     #         data_dict=data_dict,
    #     #         config=self.config,
    #     #         reference=self.reference,
    #     #         use_lang_classifier=self.use_lang_classifier,
    #     #         mem_hash=mem_hash,
    #     #         final_output=final_output
    #     #     )
    #     #     # dump
    #     #     self._running_log["lang_acc"] = data_dict["lang_acc"].item()
    #     #     self._running_log["ref_acc"] = np.mean(data_dict["ref_acc"])
    #     #     if not USE_GT:
    #     #         self._running_log["obj_acc"] = data_dict["obj_acc"].item()
    #     #         self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
    #     #         self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
    #     #         self._running_log["objectness_loss"] = data_dict["objectness_loss"]
    #     #         self._running_log["vote_loss"] = data_dict["vote_loss"]
    #     #         self._running_log["box_loss"] = data_dict["box_loss"]
    #     #         self._running_log["obj_acc"] = data_dict["obj_acc"].item()
    #     #         self._running_log["pos_ratio"] = data_dict["pos_ratio"].item()
    #     #         self._running_log["neg_ratio"] = data_dict["neg_ratio"].item()
    #     #     self._running_log["iou_rate_0.25"] = np.mean(data_dict["ref_iou_rate_0.25"])
    #     #     self._running_log["iou_rate_0.5"] = np.mean(data_dict["ref_iou_rate_0.5"])
    #     #     self._running_log["max_iou_rate_0.25"] = np.mean(data_dict["max_iou_rate_0.25"])
    #     #     self._running_log["max_iou_rate_0.5"] = np.mean(data_dict["max_iou_rate_0.5"])
    #     # else:
    #     #     self._running_log["lang_acc"] = 0
    #     #     self._running_log["ref_acc"] = 0
    #     #     self._running_log["obj_acc"] = 0
    #     #     self._running_log["pos_ratio"] = 0
    #     #     self._running_log["neg_ratio"] = 0
    #     #     self._running_log["iou_rate_0.25"] = 0
    #     #     self._running_log["iou_rate_0.5"] = 0
    #     #     self._running_log["max_iou_rate_0.25"] = 0
    #     #     self._running_log["max_iou_rate_0.5"] = 0

    # def _eval(self, phase, epoch):
    #     if self.caption and phase != "train" and epoch >= self.num_ground_epoch:
    #         bleu, cider, rouge, meteor = eval_cap(
    #             model=self.model,
    #             device=self.device,
    #             dataset=self.dataset["eval"]["val_scene"],
    #             dataloader=self.dataloader["eval"]["val_scene"],
    #             phase=phase,
    #             folder=self.stamp,
    #             use_tf=False,
    #             max_len=CONF.TRAIN.MAX_DES_LEN,
    #             force=True,
    #             min_iou=CONF.EVAL.MIN_IOU_THRESHOLD
    #         )
    #
    #         # dump
    #         self.log[phase]["bleu-1"] = bleu[0][0]
    #         self.log[phase]["bleu-2"] = bleu[0][1]
    #         self.log[phase]["bleu-3"] = bleu[0][2]
    #         self.log[phase]["bleu-4"] = bleu[0][3]
    #         self.log[phase]["cider"] = cider[0]
    #         self.log[phase]["rouge"] = rouge[0]
    #         self.log[phase]["meteor"] = meteor[0]
    #     else:
    #         self.log[phase]["bleu-1"] = 0
    #         self.log[phase]["bleu-2"] = 0
    #         self.log[phase]["bleu-3"] = 0
    #         self.log[phase]["bleu-4"] = 0
    #         self.log[phase]["cider"] = 0
    #         self.log[phase]["rouge"] = 0
    #         self.log[phase]["meteor"] = 0

    def _feed(self, dataloader, phase, epoch_id, is_eval=False):
        # switch mode
        if is_eval:
            self._set_phase("val")
        else:
            self._set_phase(phase)

        # if phase == "val" or epoch_id == 0 or not is_eval:
        #     # re-init log
        #     self._reset_log(phase)

        # change dataloader
        dataloader = dataloader if phase == "train" else tqdm(dataloader)
        """
        total_num = 1000
        self._set_phase("val")
        for data_dict in dataloader:
            # move to cuda
            for key in data_dict:
                data_dict[key] = data_dict[key].cuda()
            start = time.time()
            with torch.no_grad():
                # forward
                data_dict["epoch"] = epoch_id
                torch.cuda.synchronize()
                for _ in range(total_num):
                    data_dict = self._forward(data_dict)
                    #self._compute_loss(data_dict)
                end = time.time()
                avg = (end - start) / total_num
                print("##############################")
                print("time", end - start)
                print("avg", avg)
                print("##############################")
        """
        # enter mode
        # start_solver = time.time()
        if not is_eval:
        #if True:
        #if (not is_eval) or phase == "val":

            for i, data_dict in enumerate(tqdm(dataloader)):
                torch.cuda.empty_cache()
                # move to cuda
                for key in data_dict:
                    # data_dict[key] = data_dict[key].cuda()
                    if key != "scene_id" and key != "eval_type" and key != "istrain":
                        data_dict[key] = data_dict[key].to(self.device)

                # initialize the running loss
                # self._running_log = {
                #     # loss
                #     "loss": 0,
                #     "ref_loss": 0,
                #     "lang_loss": 0,
                #     "cap_loss": 0,
                #     "ori_loss": 0,
                #     "dist_loss": 0,
                #     "objectness_loss": 0,
                #     "vote_loss": 0,
                #     "box_loss": 0,
                #     # acc
                #     "lang_acc": 0,
                #     "ref_acc": 0,
                #     "cap_acc": 0,
                #     "ori_acc": 0,
                #     "obj_acc": 0,
                #     "pred_ious": 0,
                #     "pos_ratio": 0,
                #     "neg_ratio": 0,
                #     "iou_rate_0.25": 0,
                #     "iou_rate_0.5": 0,
                #     "max_iou_rate_0.25": 0,
                #     "max_iou_rate_0.5": 0,
                #     "scanrefer++_overall_25": 0,
                #     "scanrefer++_overall_50": 0
                # }

                # load
                # self.log[phase]["fetch"].append(0)

                # with torch.autograd.set_detect_anomaly(True):
                # forward
                data_dict["epoch"] = epoch_id
                # start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                # self.log[phase]["forward"].append(time.time() - start)

                # backward
                if phase == "train" and not is_eval:
                    # start = time.time()
                    self._backward(data_dict["loss"])
                    # self.log[phase]["backward"].append(time.time() - start)

                if i % 50 == 0:
                    wandb.log({f"{phase}_loss/total_loss": data_dict["loss"]})
                    wandb.log({f"{phase}_loss/ref_loss": data_dict["ref_loss"]})
                    wandb.log({f"{phase}_loss/lang_loss": data_dict["lang_loss"]})

                # eval
                # start = time.time()
                # self._eval(data_dict)
                # self._ground_eval(data_dict, phase, is_eval)
                # self.log[phase]["eval"].append(time.time() - start)

                # record log
                # self.log[phase]["loss"].append(self._running_log["loss"].item())
                # self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
                # self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
                # self.log[phase]["cap_loss"].append(self._running_log["cap_loss"].item())
                # self.log[phase]["ori_loss"].append(self._running_log["ori_loss"].item())
                # self.log[phase]["dist_loss"].append(self._running_log["dist_loss"].item())
                # if not USE_GT:
                #     self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
                #     self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
                #     self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
                #     self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
                #     self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
                #     self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
                #
                # self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
                # self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
                # self.log[phase]["cap_acc"].append(self._running_log["cap_acc"])
                # self.log[phase]["ori_acc"].append(self._running_log["ori_acc"])
                #
                # self.log[phase]["pred_ious"].append(self._running_log["pred_ious"])
                #
                # self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
                # self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])
                # self.log[phase]["max_iou_rate_0.25"].append(self._running_log["max_iou_rate_0.25"])
                # self.log[phase]["max_iou_rate_0.5"].append(self._running_log["max_iou_rate_0.5"])
                #
                # # report
                # if phase == "train" and not is_eval:
                #     iter_time = self.log[phase]["fetch"][-1]
                #     iter_time += self.log[phase]["forward"][-1]
                #     iter_time += self.log[phase]["backward"][-1]
                #     iter_time += self.log[phase]["eval"][-1]
                #     real_time = time.time() - start_solver
                #     self.log[phase]["real_time"].append(real_time)
                #     start_solver = time.time()
                #     self.log[phase]["iter_time"].append(iter_time)
                #     if (self._global_iter_id + 1) % self.verbose == 0:
                #         self._train_report(epoch_id)
                #
                #     # dump log
                #     if self._global_iter_id % 50 == 0:
                #         self._dump_log("train")
                #     #if self._global_iter_id != 0: self._dump_log("train")
                #     self._global_iter_id += 1


        else:
        #if is_eval:

            # self._eval(phase, epoch_id)
            if phase == "val":
                # if SCANREFER_ENHANCE:
                #     final_output = {}
                #     mem_hash = {}
                total_pred = {}
                total_gt = {}
                best_model_f1 = 0
                for data_dict in tqdm(dataloader):
                    torch.cuda.empty_cache()
                    # move to cuda
                    for key in data_dict:
                        # data_dict[key] = data_dict[key].cuda()
                        if key != "scene_id" and key != "eval_type" and key != "istrain":
                            data_dict[key] = data_dict[key].to(self.device)
                    # scanrefer++ support
                    # if SCANREFER_ENHANCE:
                    #     for scene_id in data_dict["scene_id"]:
                    #         if scene_id not in final_output:
                    #             final_output[scene_id] = []
                    # initialize the running loss
                    # self._running_log = {
                    #     # loss
                    #     "loss": 0,
                    #     "ref_loss": 0,
                    #     "lang_loss": 0,
                    #     "objectness_loss": 0,
                    #     "vote_loss": 0,
                    #     "box_loss": 0,
                    #     # acc
                    #     "lang_acc": 0,
                    #     "ref_acc": 0,
                    #     "obj_acc": 0,
                    #     "pos_ratio": 0,
                    #     "neg_ratio": 0,
                    #     "iou_rate_0.25": 0,
                    #     "iou_rate_0.5": 0,
                    #     "max_iou_rate_0.25": 0,
                    #     "max_iou_rate_0.5": 0
                    # }

                    data_dict["epoch"] = epoch_id
                    data_dict = self._forward(data_dict)
                    # self._compute_loss(data_dict)
                    pred, gt = get_eval_multi3drefer(data_dict, self.config)
                    total_pred.update(pred)
                    total_gt.update(gt)
                    # eval
                    # self._ground_eval(data_dict, phase, is_eval, mem_hash=mem_hash, final_output=final_output)

                    # if phase == "val":
                    #     # record log
                    #     self.log[phase]["loss"].append(self._running_log["loss"].item())
                    #     self.log[phase]["ref_loss"].append(self._running_log["ref_loss"].item())
                    #     self.log[phase]["lang_loss"].append(self._running_log["lang_loss"].item())
                    #     if not USE_GT:
                    #         self.log[phase]["objectness_loss"].append(self._running_log["objectness_loss"].item())
                    #         self.log[phase]["vote_loss"].append(self._running_log["vote_loss"].item())
                    #         self.log[phase]["box_loss"].append(self._running_log["box_loss"].item())
                    #         self.log[phase]["obj_acc"].append(self._running_log["obj_acc"])
                    #         self.log[phase]["pos_ratio"].append(self._running_log["pos_ratio"])
                    #         self.log[phase]["neg_ratio"].append(self._running_log["neg_ratio"])
                    #     else:
                    #         self.log[phase]["objectness_loss"].append(0)
                    #         self.log[phase]["vote_loss"].append(0)
                    #         self.log[phase]["box_loss"].append(0)
                    #         self.log[phase]["obj_acc"].append(0)
                    #         self.log[phase]["pos_ratio"].append(0)
                    #         self.log[phase]["neg_ratio"].append(0)
                    #
                    #     self.log[phase]["lang_acc"].append(self._running_log["lang_acc"])
                    #     self.log[phase]["ref_acc"].append(self._running_log["ref_acc"])
                    #
                    #
                    #     self.log[phase]["iou_rate_0.25"].append(self._running_log["iou_rate_0.25"])
                    #     self.log[phase]["iou_rate_0.5"].append(self._running_log["iou_rate_0.5"])
                    #     self.log[phase]["max_iou_rate_0.25"].append(self._running_log["max_iou_rate_0.25"])
                    #     self.log[phase]["max_iou_rate_0.5"].append(self._running_log["max_iou_rate_0.5"])

                # scanrefer+= support
                # if SCANREFER_ENHANCE and phase == "val":
                #     dir_name = f"scanrefer++_test_{SCANREFER_ENHANCE_LOSS_THRESHOLD}_{SCANREFER_ENHANCE_EVAL_THRESHOLD}_{SCANREFER_ENHANCE_VANILLE}_{USE_GT}"
                #     for key, value in final_output.items():
                #         for query in value:
                #             query["aabbs"] = [item.tolist() for item in query["aabbs"]]
                #         os.makedirs(dir_name, exist_ok=True)
                #         with open(f"{dir_name}/{key}.json", "w") as f:
                #             json.dump(value, f)
                #
                #     all_preds, all_gts = load_gt_and_pred_jsons_from_disk(dir_name, "3dvg_gt")
                #     iou_25_results, iou_50_results = evaluate_all_scenes(all_preds, all_gts)
                #     self.log[phase]["scanrefer++_overall_25"] = iou_25_results["overall"]
                #     self.log[phase]["scanrefer++_overall_50"] = iou_50_results["overall"]
                #     shutil.rmtree(dir_name)
                self.evaluator.set_ground_truths(total_gt)
                results = self.evaluator.evaluate(total_pred)

                # log
                for metric_name, result in results.items():
                    for breakdown, value in result.items():
                        wandb.log({f"val_eval/{metric_name}_{breakdown}": value})
                        if metric_name == "f1@0.5" and breakdown == "overall":
                            if value >= best_model_f1:
                                best_model_f1 = value
                                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

            # cur_criterion = self.criterion
            # if phase == "val" and cur_criterion == "sum":
            #     #metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
            #     metrics = ["bleu-4", "cider", "rouge", "meteor"]
            #     caption_cur_best = np.sum([np.mean(self.log[phase][m]) for m in metrics])
            #     ground_metrics = ["iou_rate_0.25", "iou_rate_0.5"]
            #     ground_cur_best = np.sum([np.mean(self.log[phase][m]) for m in ground_metrics])
            #     cur_best = caption_cur_best + ground_cur_best * 2
            #     if SCANREFER_ENHANCE:
            #         cur_criterion = "scanrefer++_overall_50"
            #         ground_cur_best = self.log[phase]["scanrefer++_overall_50"]
            #         cur_best = self.log[phase]["scanrefer++_overall_50"]
            # else:
            #     #cur_best = np.mean(self.log[phase][cur_criterion])
            #     caption_cur_best = 0.
            #     ground_cur_best = 0.
            #     cur_best = 0.

            # if phase == "val" and cur_best > self.best[cur_criterion]:
            #     self._log("best {} achieved: {}".format(cur_criterion, cur_best))
            #
            #     self.best["epoch"] = epoch_id + 1
            #     self.best["bleu-1"] = self.log[phase]["bleu-1"]
            #     self.best["bleu-2"] = self.log[phase]["bleu-2"]
            #     self.best["bleu-3"] = self.log[phase]["bleu-3"]
            #     self.best["bleu-4"] = self.log[phase]["bleu-4"]
            #     self.best["cider"] = self.log[phase]["cider"]
            #     self.best["rouge"] = self.log[phase]["rouge"]
            #     self.best["meteor"] = self.log[phase]["meteor"]
            #     self.best["ref_acc"] = np.mean(self.log[phase]["ref_acc"])
            #     self.best["obj_acc"] = np.mean(self.log[phase]["obj_acc"])
            #     self.best["pos_ratio"] = np.mean(self.log[phase]["pos_ratio"])
            #     self.best["neg_ratio"] = np.mean(self.log[phase]["neg_ratio"])
            #     self.best["iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
            #     self.best["iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])
            #     self.best["sum"] = cur_best
            #     self.best["scanrefer++_overall_25"] = self.log[phase]["scanrefer++_overall_25"]
            #     self.best["scanrefer++_overall_50"] = self.log[phase]["scanrefer++_overall_50"]
            #
            #     # save model
            #     self._log("saving best models...\n")
            #     model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            #     torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))
            #
            # if phase == "val" and caption_cur_best > self.best["caption_sum"]:
            #     self._log("best caption {} achieved: {}".format(cur_criterion, caption_cur_best))
            #     self.best["best_caption_epoch"] = epoch_id + 1
            #     self.best["best_caption_bleu-4"] = self.log[phase]["bleu-4"]
            #     self.best["best_caption_cider"] = self.log[phase]["cider"]
            #     self.best["best_caption_rouge"] = self.log[phase]["rouge"]
            #     self.best["best_caption_meteor"] = self.log[phase]["meteor"]
            #     self.best["caption_sum"] = caption_cur_best
            #
            #     # save model
            #     self._log("saving best caption models...\n")
            #     model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            #     torch.save(self.model.state_dict(), os.path.join(model_root, "caption_model.pth"))
            #
            # if phase == "val" and ground_cur_best > self.best["ground_sum"]:
            #     self._log("best ground {} achieved: {}".format(cur_criterion, ground_cur_best))
            #
            #     self.best["best_ground_epoch"] = epoch_id + 1
            #     self.best["best_ground_iou_rate_0.25"] = np.mean(self.log[phase]["iou_rate_0.25"])
            #     self.best["best_ground_iou_rate_0.5"] = np.mean(self.log[phase]["iou_rate_0.5"])
            #     self.best["ground_sum"] = ground_cur_best
            #     self.best["scanrefer++_overall_25"] = self.log[phase]["scanrefer++_overall_25"]
            #     self.best["scanrefer++_overall_50"] = self.log[phase]["scanrefer++_overall_50"]
            #
            #     # save model
            #     self._log("saving best ground models...\n")
            #     model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
            #     torch.save(self.model.state_dict(), os.path.join(model_root, "ground_model.pth"))

    # def _finish(self, epoch_id):
    #     # print best
    #     self._best_report()
    #
    #     # save check point
    #     self._log("saving checkpoint...\n")
    #     save_dict = {
    #         "epoch": epoch_id,
    #         "model_state_dict": self.model.state_dict(),
    #         "optimizer_state_dict": self.optimizer.state_dict(),
    #         "best": self.best
    #     }
    #     checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
    #     torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))
    #
    #     # save model
    #     self._log("saving last models...\n")
    #     model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
    #     torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))
    #
    #     # export
    #     for phase in ["train", "val"]:
    #         self._log_writer[phase].export_scalars_to_json(os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))
    #
    # def _train_report(self, epoch_id):
    #     # compute ETA
    #     fetch_time = self.log["train"]["fetch"]
    #     forward_time = self.log["train"]["forward"]
    #     backward_time = self.log["train"]["backward"]
    #     eval_time = self.log["train"]["eval"]
    #     iter_time = self.log["train"]["iter_time"]
    #     real_time = self.log["train"]["real_time"]
    #
    #     mean_train_time = np.mean(iter_time)
    #     mean_train_time = np.mean(real_time)
    #     mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])
    #
    #     num_train_iter_left = self._total_iter["train"] - self._global_iter_id - 1
    #     eta_sec = num_train_iter_left * mean_train_time
    #
    #     num_val_times = num_train_iter_left // self.val_step
    #     eta_sec += len(self.dataloader["eval"]["train"]) * num_val_times * mean_est_val_time
    #     eta_sec += len(self.dataloader["eval"]["val"]) * num_val_times * mean_est_val_time
    #
    #     eta = decode_eta(eta_sec)
    #
    #     # print report
    #     if not USE_GT:
    #         iter_report = self.__iter_report_template.format(
    #             epoch_id=epoch_id + 1,
    #             iter_id=self._global_iter_id + 1,
    #             total_iter=self._total_iter["train"],
    #             train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
    #             train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
    #             train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
    #             train_cap_loss=round(np.mean([v for v in self.log["train"]["cap_loss"]]), 5),
    #             train_ori_loss=round(np.mean([v for v in self.log["train"]["ori_loss"]]), 5),
    #             train_dist_loss=round(np.mean([v for v in self.log["train"]["dist_loss"]]), 5),
    #             train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
    #             train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
    #             train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
    #             train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
    #             train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
    #             train_cap_acc=round(np.mean([v for v in self.log["train"]["cap_acc"]]), 5),
    #             train_ori_acc=round(np.mean([v for v in self.log["train"]["ori_acc"]]), 5),
    #             train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
    #             train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
    #             train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
    #             train_iou_max_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
    #             train_iou_max_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),
    #             train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
    #             train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
    #             train_pred_ious=round(np.mean([v for v in self.log["train"]["pred_ious"]]), 5),
    #             mean_fetch_time=round(np.mean(fetch_time), 5),
    #             mean_forward_time=round(np.mean(forward_time), 5),
    #             mean_backward_time=round(np.mean(backward_time), 5),
    #             mean_eval_time=round(np.mean(eval_time), 5),
    #             mean_iter_time=round(np.mean(iter_time), 5),
    #             mean_real_time=round(np.mean(real_time), 5),
    #             eta_h=eta["h"],
    #             eta_m=eta["m"],
    #             eta_s=eta["s"]
    #         )
    #     else:
    #         iter_report = self.__iter_report_template.format(
    #             epoch_id=epoch_id + 1,
    #             iter_id=self._global_iter_id + 1,
    #             total_iter=self._total_iter["train"],
    #             train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
    #             train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
    #             train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
    #             train_cap_loss=round(np.mean([v for v in self.log["train"]["cap_loss"]]), 5),
    #             train_ori_loss=0,
    #             train_dist_loss=round(np.mean([v for v in self.log["train"]["dist_loss"]]), 5),
    #             train_objectness_loss=0,
    #             train_vote_loss=0,
    #             train_box_loss=0,
    #             train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
    #             train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
    #             train_cap_acc=round(np.mean([v for v in self.log["train"]["cap_acc"]]), 5),
    #             train_ori_acc=0,
    #             train_obj_acc=0,
    #             train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
    #             train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
    #             train_iou_max_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
    #             train_iou_max_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),
    #             train_pos_ratio=0,
    #             train_neg_ratio=0,
    #             train_pred_ious=0,
    #             mean_fetch_time=round(np.mean(fetch_time), 5),
    #             mean_forward_time=round(np.mean(forward_time), 5),
    #             mean_backward_time=round(np.mean(backward_time), 5),
    #             mean_eval_time=round(np.mean(eval_time), 5),
    #             mean_iter_time=round(np.mean(iter_time), 5),
    #             mean_real_time=round(np.mean(real_time), 5),
    #             eta_h=eta["h"],
    #             eta_m=eta["m"],
    #             eta_s=eta["s"]
    #         )
    #     self._log(iter_report)
    #
    # def _epoch_report(self, epoch_id):
    #     self._log("epoch [{}/{}] done...".format(epoch_id+1, self.epoch))
    #     self._log_eval("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))
    #     if not USE_GT:
    #         epoch_report = self.__epoch_report_template.format(
    #             train_bleu_1=0,
    #             train_bleu_2=0,
    #             train_bleu_3=0,
    #             train_bleu_4=0,
    #             train_cider=0,
    #             train_rouge=0,
    #             train_meteor=0,
    #             train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
    #             train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
    #             train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
    #             train_objectness_loss=round(np.mean([v for v in self.log["train"]["objectness_loss"]]), 5),
    #             train_vote_loss=round(np.mean([v for v in self.log["train"]["vote_loss"]]), 5),
    #             train_box_loss=round(np.mean([v for v in self.log["train"]["box_loss"]]), 5),
    #             train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
    #             train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
    #             train_obj_acc=round(np.mean([v for v in self.log["train"]["obj_acc"]]), 5),
    #             train_pos_ratio=round(np.mean([v for v in self.log["train"]["pos_ratio"]]), 5),
    #             train_neg_ratio=round(np.mean([v for v in self.log["train"]["neg_ratio"]]), 5),
    #             train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
    #             train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
    #             train_max_iou_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
    #             train_max_iou_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),
    #             val_bleu_1=round(self.log["val"]["bleu-1"], 5),
    #             val_bleu_2=round(self.log["val"]["bleu-2"], 5),
    #             val_bleu_3=round(self.log["val"]["bleu-3"], 5),
    #             val_bleu_4=round(self.log["val"]["bleu-4"], 5),
    #             val_cider=round(self.log["val"]["cider"], 5),
    #             val_rouge=round(self.log["val"]["rouge"], 5),
    #             val_meteor=round(self.log["val"]["meteor"], 5),
    #             val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
    #             val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
    #             val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
    #             val_objectness_loss=round(np.mean([v for v in self.log["val"]["objectness_loss"]]), 5),
    #             val_vote_loss=round(np.mean([v for v in self.log["val"]["vote_loss"]]), 5),
    #             val_box_loss=round(np.mean([v for v in self.log["val"]["box_loss"]]), 5),
    #             val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
    #             val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
    #             val_obj_acc=round(np.mean([v for v in self.log["val"]["obj_acc"]]), 5),
    #             val_pos_ratio=round(np.mean([v for v in self.log["val"]["pos_ratio"]]), 5),
    #             val_neg_ratio=round(np.mean([v for v in self.log["val"]["neg_ratio"]]), 5),
    #             val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
    #             val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
    #             val_max_iou_rate_25=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.25"]]), 5),
    #             val_max_iou_rate_5=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.5"]]), 5),
    #             scanrefer_plus_overall_25=self.log["val"]["scanrefer++_overall_25"],
    #             scanrefer_plus_overall_50=self.log["val"]["scanrefer++_overall_50"],
    #         )
    #     else:
    #         epoch_report = self.__epoch_report_template.format(
    #             train_bleu_1=0,
    #             train_bleu_2=0,
    #             train_bleu_3=0,
    #             train_bleu_4=0,
    #             train_cider=0,
    #             train_rouge=0,
    #             train_meteor=0,
    #             train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
    #             train_ref_loss=round(np.mean([v for v in self.log["train"]["ref_loss"]]), 5),
    #             train_lang_loss=round(np.mean([v for v in self.log["train"]["lang_loss"]]), 5),
    #             train_objectness_loss=0,
    #             train_vote_loss=0,
    #             train_box_loss=0,
    #             train_lang_acc=round(np.mean([v for v in self.log["train"]["lang_acc"]]), 5),
    #             train_ref_acc=round(np.mean([v for v in self.log["train"]["ref_acc"]]), 5),
    #             train_obj_acc=0,
    #             train_pos_ratio=0,
    #             train_neg_ratio=0,
    #             train_iou_rate_25=round(np.mean([v for v in self.log["train"]["iou_rate_0.25"]]), 5),
    #             train_iou_rate_5=round(np.mean([v for v in self.log["train"]["iou_rate_0.5"]]), 5),
    #             train_max_iou_rate_25=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.25"]]), 5),
    #             train_max_iou_rate_5=round(np.mean([v for v in self.log["train"]["max_iou_rate_0.5"]]), 5),
    #             val_bleu_1=round(self.log["val"]["bleu-1"], 5),
    #             val_bleu_2=round(self.log["val"]["bleu-2"], 5),
    #             val_bleu_3=round(self.log["val"]["bleu-3"], 5),
    #             val_bleu_4=round(self.log["val"]["bleu-4"], 5),
    #             val_cider=round(self.log["val"]["cider"], 5),
    #             val_rouge=round(self.log["val"]["rouge"], 5),
    #             val_meteor=round(self.log["val"]["meteor"], 5),
    #             val_loss=round(np.mean([v for v in self.log["val"]["loss"]]), 5),
    #             val_ref_loss=round(np.mean([v for v in self.log["val"]["ref_loss"]]), 5),
    #             val_lang_loss=round(np.mean([v for v in self.log["val"]["lang_loss"]]), 5),
    #             val_objectness_loss=0,
    #             val_vote_loss=0,
    #             val_box_loss=0,
    #             val_lang_acc=round(np.mean([v for v in self.log["val"]["lang_acc"]]), 5),
    #             val_ref_acc=round(np.mean([v for v in self.log["val"]["ref_acc"]]), 5),
    #             val_obj_acc=0,
    #             val_pos_ratio=0,
    #             val_neg_ratio=0,
    #             val_iou_rate_25=round(np.mean([v for v in self.log["val"]["iou_rate_0.25"]]), 5),
    #             val_iou_rate_5=round(np.mean([v for v in self.log["val"]["iou_rate_0.5"]]), 5),
    #             val_max_iou_rate_25=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.25"]]), 5),
    #             val_max_iou_rate_5=round(np.mean([v for v in self.log["val"]["max_iou_rate_0.5"]]), 5),
    #             scanrefer_plus_overall_25=self.log["val"]["scanrefer++_overall_25"],
    #             scanrefer_plus_overall_50=self.log["val"]["scanrefer++_overall_50"],
    #         )
    #     self._log(epoch_report)
    #     self._log_eval(epoch_report)
    #
    # def _best_report(self):
    #     self._log("training completed...")
    #     if not USE_GT:
    #         best_report = self.__best_report_template.format(
    #             epoch=self.best["epoch"],
    #             bleu_1=round(self.best["bleu-1"], 5),
    #             bleu_2=round(self.best["bleu-2"], 5),
    #             bleu_3=round(self.best["bleu-3"], 5),
    #             bleu_4=round(self.best["bleu-4"], 5),
    #             cider=round(self.best["cider"], 5),
    #             rouge=round(self.best["rouge"], 5),
    #             meteor=round(self.best["meteor"], 5),
    #             ref_acc=round(self.best["ref_acc"], 5),
    #             obj_acc=round(self.best["obj_acc"], 5),
    #             pos_ratio=round(self.best["pos_ratio"], 5),
    #             neg_ratio=round(self.best["neg_ratio"], 5),
    #             iou_rate_25=round(self.best["iou_rate_0.25"], 5),
    #             iou_rate_5=round(self.best["iou_rate_0.5"], 5),
    #             best_caption_epoch=self.best["best_caption_epoch"],
    #             best_caption_bleu_4=round(self.best["best_caption_bleu-4"], 5),
    #             best_caption_cider=round(self.best["best_caption_cider"], 5),
    #             best_caption_rouge=round(self.best["best_caption_rouge"], 5),
    #             best_caption_meteor=round(self.best["best_caption_meteor"], 5),
    #             best_ground_epoch=self.best["best_ground_epoch"],
    #             best_ground_iou_rate_25=round(self.best["best_ground_iou_rate_0.25"], 5),
    #             best_ground_iou_rate_5=round(self.best["best_ground_iou_rate_0.5"], 5),
    #         )
    #     else:
    #         best_report = self.__best_report_template.format(
    #             epoch=self.best["epoch"],
    #             bleu_1=round(self.best["bleu-1"], 5),
    #             bleu_2=round(self.best["bleu-2"], 5),
    #             bleu_3=round(self.best["bleu-3"], 5),
    #             bleu_4=round(self.best["bleu-4"], 5),
    #             cider=round(self.best["cider"], 5),
    #             rouge=round(self.best["rouge"], 5),
    #             meteor=round(self.best["meteor"], 5),
    #             ref_acc=round(self.best["ref_acc"], 5),
    #             obj_acc=0,
    #             pos_ratio=0,
    #             neg_ratio=0,
    #             iou_rate_25=round(self.best["iou_rate_0.25"], 5),
    #             iou_rate_5=round(self.best["iou_rate_0.5"], 5),
    #             best_caption_epoch=self.best["best_caption_epoch"],
    #             best_caption_bleu_4=round(self.best["best_caption_bleu-4"], 5),
    #             best_caption_cider=round(self.best["best_caption_cider"], 5),
    #             best_caption_rouge=round(self.best["best_caption_rouge"], 5),
    #             best_caption_meteor=round(self.best["best_caption_meteor"], 5),
    #             best_ground_epoch=self.best["best_ground_epoch"],
    #             best_ground_iou_rate_25=round(self.best["best_ground_iou_rate_0.25"], 5),
    #             best_ground_iou_rate_5=round(self.best["best_ground_iou_rate_0.5"], 5),
    #         )
    #     self._log(best_report)
    #     with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
    #         f.write(best_report)
