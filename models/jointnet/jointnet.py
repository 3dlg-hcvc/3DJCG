import torch
import torch.nn as nn
from macro import *
from models.single_obj_encoder import PointNetPP
from models.base_module.backbone_module import Pointnet2Backbone
from models.base_module.voting_module import VotingModule
from models.base_module.lang_module import LangModule
from models.proposal_module.proposal_module_fcos import ProposalModule
from models.proposal_module.relation_module import RelationModule
from models.refnet.match_module import MatchModule
# from models.capnet.caption_module import SceneCaptionModule, TopDownSceneCaptionModule

class JointNet(nn.Module):
    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, vocabulary, embeddings,
                 input_feature_dim=0, num_proposal=128, num_locals=-1, vote_factor=1, sampling="vote_fps",
                 no_caption=False, use_topdown=False, query_mode="corner", num_graph_steps=0, use_relation=False,
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, ground_hidden_size=256, caption_hidden_size=512, dataset_config=None):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.no_caption = no_caption
        self.dataset_config = dataset_config

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        if not USE_GT:
            self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

            # Hough voting
            self.vgen = VotingModule(self.vote_factor, 256)
        else:
            self.single_obj_encoder = PointNetPP(
                sa_n_points=[32, 16, None], sa_n_samples=[32, 32, None],
                sa_radii=[0.2, 0.4, None], sa_mlps=[[132, 64, 64, 128], [128, 128, 128, 256],[256, 256, 512, 128]]
            )
        self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal,
                                       sampling, config=self.dataset_config)
        self.relation = RelationModule(num_proposals=num_proposal, det_channel=128)  # bef 256


        # Vote aggregation and object proposal

        if not no_reference:
            # --------- LANGUAGE ENCODING ---------
            # Encode the input descriptions into vectors
            # (including attention and language classification)
            self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size, ground_hidden_size)

            # --------- PROPOSAL MATCHING ---------
            # Match the generated proposals and select the most confident ones
            self.match = MatchModule(num_proposals=num_proposal, lang_size=(1 + int(self.use_bidir)) * ground_hidden_size, det_channel=128)  # bef 256

        # if not no_caption:
        #     if use_topdown:
        #         self.caption = TopDownSceneCaptionModule(vocabulary, embeddings, emb_size, 128,
        #             caption_hidden_size, num_proposal, num_locals, query_mode, use_relation)
        #     else:
        #         self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, caption_hidden_size, num_proposal)


    def forward(self, data_dict, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds,
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        # --------- HOUGH VOTING ---------
        if not USE_GT:
            data_dict = self.backbone_net(data_dict)

            # --------- HOUGH VOTING ---------
            xyz = data_dict["fp2_xyz"]
            features = data_dict["fp2_features"]
            data_dict["seed_inds"] = data_dict["fp2_inds"]
            data_dict["seed_xyz"] = xyz
            data_dict["seed_features"] = features

            xyz, features = self.vgen(data_dict, xyz, features)
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            data_dict["vote_xyz"] = xyz
        else:
            features = self.single_obj_encoder(data_dict["single_obj_info"].flatten(0, 1))
            features = features.reshape(len(data_dict["scene_id"]), -1, features.shape[-1])
            features_norm = torch.norm(features, p=2, dim=1)
            features = features.div(features_norm.unsqueeze(1))
            xyz = data_dict['center_label'][:, :, 0:3]


        # data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)



        data_dict = self.relation(data_dict)

        if not self.no_reference:
            #######################################
            #                                     #
            #           LANGUAGE BRANCH           #
            #                                     #
            #######################################

            # --------- LANGUAGE ENCODING ---------
            data_dict = self.lang(data_dict)

            #######################################
            #                                     #
            #          PROPOSAL MATCHING          #
            #                                     #
            #######################################

            # --------- PROPOSAL MATCHING ---------
            # config for bbox_embedding
            data_dict = self.match(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        # if not self.no_caption:
        #     data_dict = self.caption(data_dict, use_tf, is_eval)

        return data_dict
