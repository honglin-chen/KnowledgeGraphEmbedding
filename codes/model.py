#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset
import pdb
import utils.hyperbolic_utils as hyperbolic
from termcolor import colored
from utils.math_utils import arsinh, arcosh
from utils import manifolds
import matplotlib.pyplot as plt
from operator import itemgetter
import os

new_dict = {}
class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, 
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        
        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        if model_name == 'PoincareEmbedding':
            pdb.set_trace()
            print('Check training input')
            self.embedding_range = nn.Parameter(
                torch.Tensor([1e-3]),
                requires_grad=False
            )
            dummy_index = nentity - 1
            self.dummy_node = dummy_index
            # self.entity_embedding.data[dummy_index, :] *= 1e-5

        if model_name == 'RotatTransH2':
            self.embedding_range = nn.Parameter(
                torch.Tensor([1e-3]),
                requires_grad=False
            )

        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        if model_name.endswith('tRotatE') or model_name.endswith('tRotationH'):
            self.n_tuple = int(model_name[0])
            assert self.n_tuple in [2, 4]
            self.entity_dim *= self.n_tuple

        if model_name == 'LinearTransE':
            self.relation_dim *= 8

        if model_name in ['RotatH', 'RotatTransH', 'expRotatTransH', 'RotatHdegree', 'RotatHR']:
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.entity_dim += 1  # use the last column as entity bias
            if model_name in ['RotatTransH', 'expRotatTransH', 'RotatHdegree', 'RotatHR']:
                self.relation_dim += (self.entity_dim-1)
                if model_name == 'RotatHR':
                    self.relation_dim += 1  # use the last column as relation-specific coefficient

                # if model_name == 'expRotatTransH':
                #     self.relation_dim += 1  # use the last column as relation-specific curvature

        if model_name == 'RotatTransH2':
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.relation_dim += self.entity_dim
            self.entity_dim += 1

        if model_name in ['RotatCones', 'RotatCones2', 'PoincareEmbedding', 'EmbedCones', 'ConeLCA']:
            pdb.set_trace()
            print('Check training input')
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.relation_dim += self.entity_dim
            self.entity_dim += 1

            self.relation_dim += 1

        if model_name == 'BoxLCA':
            pdb.set_trace()
            print('Check training input')
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.relation_dim += self.entity_dim
            self.entity_dim *= 2
            self.relation_dim += 1
            self.center_alpha = 0.2


        if model_name in ['RotatCones3']:
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.relation_dim += hidden_dim
            self.entity_dim += 1


        if model_name == 'RotatTransE':
            self.relation_dim += self.entity_dim

        self.model_name = model_name


        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))

        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'BoxLCA':
            nn.init.uniform_(
                tensor=self.entity_embedding.data[:, int(self.entity_dim/2):],
                a=0.,
                b=self.embedding_range.item()
            )
            print(self.entity_embedding)
            dummy_index = nentity - 1
            self.dummy_node = dummy_index
            self.entity_embedding.data[dummy_index, :] *= 1e-9


        # Initialize outside of the K ball, but inside the unit ball.

        if model_name == 'RotatTransH2':
            poincare_init_path = './models/PoincareEmbedding_synthetic_tree_1_-1/checkpoint/ckpt_7999'
            pdb.set_trace()

            ckpt = torch.load(poincare_init_path)

            # adjust model state dict:

            self.load_state_dict(ckpt['model_state_dict'])
            print('Init checkpoint from %s' % poincare_init_path)

            resc_vecs = 0.5

            temp_e = self.entity_embedding.data
            temp_r = self.relation_embedding.data

            with torch.no_grad():
                entity_embedding_data = self.entity_embedding.data.clone()
                translation_embedding_data = self.relation_embedding.data.clone()[:, -self.entity_dim:]
                c = self.softplus(self.curvature)

                entity_embedding_data = hyperbolic.expmap0(entity_embedding_data, c)
                translation_embedding_data = hyperbolic.expmap0(translation_embedding_data, c)

                entity_embedding_data *= resc_vecs
                translation_embedding_data *= resc_vecs

                entity_embedding_data = hyperbolic.logmap0(entity_embedding_data, c)
                translation_embedding_data = hyperbolic.logmap0(translation_embedding_data, c)
                self.entity_embedding.data = entity_embedding_data
                self.relation_embedding.data[:, -self.entity_dim:] = translation_embedding_data

        if model_name in ['RotatCones', 'RotatCones2', 'RotatCones3', 'EmbedCones', 'ConeLCA']:

            # Note: small initialization is important for the model to work well:
            # So far, the best parameters are:
            # K = 0.1, EPS = 1e-3
            # embedding_min = self.inner_radius + EPS
            # embedding_max = self.inner_radius + 2 * EPS

            K = 0.1
            self.K = K
            self.EPS = 1e-3
            self.inner_radius = 2 * K / (1 + np.sqrt(1 + 4 * K * K))
            self.manifold = manifolds.PoincareManifold(K=K)

            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            #
            self.embedding_range = nn.Parameter(torch.Tensor([self.inner_radius + self.EPS * 2]), requires_grad=False)
            # self.embedding_range = nn.Parameter(torch.Tensor([1e-2]), requires_grad=False)
            # self.embedding_range = nn.Parameter(
            #     torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            #     requires_grad=False
            # )
            nn.init.uniform_(tensor=self.entity_embedding,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.relation_embedding,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())

            print('Warning: set relation_embedding last colum zero')
            self.relation_embedding.data[:, -1:] = 0.
            print(self.relation_embedding)

            self.sigmoid = torch.nn.Sigmoid()

            with torch.no_grad():
                true_entity_dim = self.entity_dim if self.entity_dim % 2 == 0 else self.entity_dim - 1
                embedding = self.entity_embedding.data[:, 0:true_entity_dim]

                embedding = torch.stack(torch.chunk(embedding, 2, dim=1), dim=2)
                norms = embedding.norm(dim=-1, keepdim=True)
                embedding = embedding / (norms + 1e-15) # unit norm
                rand_norm = torch.empty(norms.shape).uniform_(self.inner_radius + self.EPS, self.inner_radius + self.EPS * 2)
                embedding *= rand_norm
                embedding = embedding.view(nentity, true_entity_dim)

                self.entity_embedding.data[:, 0:true_entity_dim] = embedding

            print('K: %.5f, embedding_range: %.5f, embedding_min: %.5f, embedding_max: %.5f' % \
                  (self.K, self.embedding_range, self.inner_radius + self.EPS, self.inner_radius + self.EPS * 2))


            self.dummy_node = nentity - 1
            self.dummy_relation = nrelation - 1
            self.entity_embedding.data[self.dummy_node, :] *= 1e-5

            # poincare_init_path = './models/PoincareEmbedding_synthetic_tree_2_-1/checkpoint/ckpt_69999'
            # poincare_init_path = './models/PoincareEmbedding_wn18-hierarchy_-1/checkpoint/ckpt_14999'
            # poincare_init_path = './models/PoincareEmbedding_wn18_hypernym_-3/checkpoint/ckpt_16999'
            poincare_init_path = './models/ConeLCA_wn18_hypernym_-1/checkpoint/ckpt_79999'

            # FIXME: hangle dummy relation
            # self.entity_embedding = nn.Parameter(torch.zeros(nentity - 1, self.entity_dim))
            # self.relation_embedding = nn.Parameter(torch.zeros(nrelation - 1, self.relation_dim))
            ckpt = torch.load(poincare_init_path)

            self.load_state_dict(ckpt['model_state_dict'])
            print('Init checkpoint from %s' % poincare_init_path)

            # # FIXME: hacky way of adjusting the relation
            # temp_relation = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            # temp_relation.data[0:nrelation-1, :] = self.relation_embedding.data
            # self.relation_embedding = temp_relation
            #
            # temp_entity = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            # temp_entity.data[0:nentity - 1, :] = self.entity_embedding.data
            # self.entity_embedding = temp_entity

            resc_vecs = 1.0

            temp_e = self.entity_embedding.data
            temp_r = self.relation_embedding.data

            with torch.no_grad():
                entity_embedding_data = self.entity_embedding.data.clone()
                translation_embedding_data = self.relation_embedding.data.clone()[:, -self.entity_dim:]
                c = self.softplus(self.curvature)

                entity_embedding_data = hyperbolic.expmap0(entity_embedding_data, c)
                translation_embedding_data = hyperbolic.expmap0(translation_embedding_data, c)

                entity_embedding_data *= resc_vecs
                translation_embedding_data *= resc_vecs

                entity_embedding_data = hyperbolic.logmap0(entity_embedding_data, c)
                translation_embedding_data = hyperbolic.logmap0(translation_embedding_data, c)
                self.entity_embedding.data = entity_embedding_data
                self.relation_embedding.data[:, -self.entity_dim:] = translation_embedding_data

            # self.root = nn.Parameter(self.entity_embedding.data[0, :], requires_grad=False)
            # self.root.data[0] += 1e-3
            # self.root.data[1] -= 1e-3

            # print('Warning: set relation_embedding last colum zero')
            # # self.relation_embedding.data[:, -1] = 0.
            print(self.relation_embedding)
            # #

            # clip vectors
            # self.entity_embedding = nn.Parameter(self.clip_vector(self.entity_embedding.data))


        if model_name == 'LinearTransE':
            # print('Warning: Using xavier uniform init for relation embedding')
            # nn.init.xavier_uniform_(self.relation_embedding)
            print(colored('!! Warning: initiaze 1st part of relation embedding to be 1 !!', 'red'))
            nn.init.ones_(self.relation_embedding[..., 0:int(self.relation_dim/2.0)])

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', '2tRotatE', '4tRotatE', 'pRotatE', \
                              'LinearTransE', 'RotatH', 'RotatTransH', 'RotatTransE', 'MobiusE',
                              'expRotatTransH', 'RotatCones', 'RotatCones2', 'PoincareEmbedding', 'RotatCones3', 'RotatTransH2',
                              'EmbedCones', 'ConeLCA', 'BoxLCA']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'tRotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('tRotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

        print('Embedding range: %.5f' % self.embedding_range.data)

    def forward(self, sample, mode='single', degree=None):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=sample[:,1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=sample[:,2]
            ).unsqueeze(1)

            if sample.shape[1] == 4:
                relation_category = sample[:, 3]
                relation = (relation, relation_category)

            
        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            
            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)
            
            relation = torch.index_select(
                self.relation_embedding, 
                dim=0, 
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part[:, 2]
            ).unsqueeze(1)

            
        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=head_part[:, 0]
            ).unsqueeze(1)
            
            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)


            tail = torch.index_select(
                self.entity_embedding, 
                dim=0, 
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            if head_part.shape[1] == 4:
                relation_category = head_part[:, 3]
                relation = (relation, relation_category)
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotatTransE': self.RotatE,
            'RotatH': self.RotatH,
            'RotatHR': self.RotatHR,
            'RotatTransH': self.RotatH,
            'RotatTransH2': self.RotatH2,
            'RotatCones': self.RotatCones,
            'RotatCones2': self.RotatCones2,
            'RotatCones3': self.RotatCones3,
            'EmbedCones': self.EmbedCones,
            'ConeLCA': self.ConeLCA,
            'BoxLCA': self.BoxLCA,
            'PoincareEmbedding': self.PoincareEmbedding
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        elif self.model_name == 'RotatHdegree':
            score = self.RotatHdegree(head, relation, tail, mode, degree)
        else:
            raise ValueError('model %s not supported' % self.model_name)
        
        return score
    
    def TransE(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim = 2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim = 2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]
        if self.model_name == 'RotatTransE':
            phase_relation, translation = relation.split([int(self.entity_dim / 2.), self.entity_dim], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            re_trans, im_trans = torch.chunk(translation, 2, dim=2)
        else:
            phase_relation = relation/(self.embedding_range.item()/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        if self.model_name == 'RotatTransE':
            re_score += re_trans
            im_score += im_trans

        # pdb.set_trace()
        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma.item() - score.sum(dim = 2)
        return score

    def RotatH(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)
        bh = bh.squeeze(2)
        bt = bt.squeeze(2)

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if self.model_name == 'RotatTransH':
            phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)


        # Warning: Adjust translation to Local Frame of Reference
        # pdb.set_trace()
        theta = self.compute_rot_angle(re_head, im_head)
        re_theta = theta.cos()
        im_theta = theta.sin()
        re_translation, im_translation = torch.chunk(translation, 2, dim=2)

        re_rot_translation = re_translation * re_theta - im_translation * im_theta
        im_rot_translation = re_translation * im_theta + im_translation * re_theta
        translation = torch.cat([re_rot_translation, im_rot_translation], dim=2)
        # pdb.set_trace()


        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
            if self.model_name == 'RotatTransH':
                res = hyperbolic.mobius_add(tail, -translation, c)
                res = hyperbolic.proj(res, c=c)
                re_tail, im_tail = torch.chunk(res, 2, dim=2)

            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            res = torch.cat([re_score, im_score], dim=2)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, head, c)

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if self.model_name == 'RotatTransH':
                res = hyperbolic.proj(res, c=c)
                res = hyperbolic.mobius_add(res, translation, c)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, tail, c)

        score = bh + bt - score.squeeze(2)
        # score = 6.0 - score.squeeze(2)
        # print(mode, head.shape, torch.mean(score).item())

        return score

    def RotatH2(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)
        bh = bh.squeeze(2)
        bt = bt.squeeze(2)

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)
        #
        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        #
        # # Make phases of relations uniformly distributed in [-pi, pi]
        if self.model_name == 'RotatTransH2':
            phase_relation, translation = relation.split([int((self.entity_dim-1) / 2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)
        #
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)
        #
        # if mode == 'head-batch':
        #     raise ValueError
        #     if self.model_name == 'RotatTransH2':
        #         res = hyperbolic.mobius_add(tail, -translation, c)
        #         res = hyperbolic.proj(res, c=c)
        #         re_tail, im_tail = torch.chunk(res, 2, dim=2)
        #
        #     re_score = re_relation * re_tail + im_relation * im_tail
        #     im_score = re_relation * im_tail - im_relation * re_tail
        #
        #     res = torch.cat([re_score, im_score], dim=2)
        #     res = hyperbolic.proj(res, c=c)
        #     score = hyperbolic.sqdist(res, head, c)
        #
        # else:
        #     re_score = re_head * re_relation - im_head * im_relation
        #     im_score = re_head * im_relation + im_head * re_relation
        #
        #     res = torch.cat([re_score, im_score], dim=2)
        #
        #     if self.model_name == 'RotatTransH2':
        #         res = hyperbolic.proj(res, c=c)
        #         res = hyperbolic.mobius_add(res, translation, c)
        #     res = hyperbolic.proj(res, c=c)
        #     score = hyperbolic.sqdist(res, tail, c)

        # score = bh + bt - score.squeeze(2)
        # score = 6.0 - score.squeeze(2)
        # print(mode, head.shape, torch.mean(score).item())
        res = hyperbolic.mobius_add(head, translation, c)
        res = hyperbolic.proj(res, c=c)
        score = 0.005 - hyperbolic.sqdist(res, tail, c).squeeze(2)

        return score

    def RotatHR(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        translation_flag = True

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)
        bh = bh.squeeze(2)
        bt = bt.squeeze(2)

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        if translation_flag:
            phase_relation, translation, coefficient = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1, 1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
            if translation_flag:
                res = hyperbolic.mobius_add(tail, -translation, c)
                res = hyperbolic.proj(res, c=c)
                re_tail, im_tail = torch.chunk(res, 2, dim=2)

            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            res = torch.cat([re_score, im_score], dim=2)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, head, c)

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if translation_flag:
                res = hyperbolic.proj(res, c=c)
                res = hyperbolic.mobius_add(res, translation, c)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, tail, c)

        score = self.softplus(coefficient.squeeze(2)) * (bh + bt) - score.squeeze(2)

        return score

    def RotatHdegree(self, head, relation, tail, mode, degree):
        pi = 3.14159265358979323846
        translation_flag = True

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)
        bh = bh.squeeze(2)
        bt = bt.squeeze(2)

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
            if translation_flag:
                res = hyperbolic.mobius_add(tail, -translation, c)
                res = hyperbolic.proj(res, c=c)
                re_tail, im_tail = torch.chunk(res, 2, dim=2)

            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail

            res = torch.cat([re_score, im_score], dim=2)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, head, c)

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if translation_flag:
                res = hyperbolic.proj(res, c=c)
                res = hyperbolic.mobius_add(res, translation, c)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, tail, c)

        q = 2

        degree_coefficient = degree ** (1. / q) # / 10.0
        score = degree_coefficient * (bh + bt) - score.squeeze(2)

        return score

    def PoincareEmbedding(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        translation_flag = True
        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        # # pdb.set_trace()
        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        #
        # # Make phases of relations uniformly distributed in [-pi, pi]
        # if translation_flag:
        #     phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
        #     phase_relation = phase_relation / (self.embedding_range.item() / pi)
        #     translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        # else:
        #     phase_relation = relation / (self.embedding_range.item() / pi)
        #
        #
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError

        else:
            pass
            # Remove Rotation
            # re_score = re_head * re_relation - im_head * im_relation
            # im_score = re_head * im_relation + im_head * re_relation
            #
            # res = torch.cat([re_score, im_score], dim=2)
            #
            # if translation_flag:
            #     res = hyperbolic.proj(res, c=c)
            #     res = hyperbolic.mobius_add(res, translation, c)
            # res = hyperbolic.proj(res, c=c)
            # score = hyperbolic.sqdist(res, tail, c)

        # res = hyperbolic.mobius_add(head, translation, c)
        # res = hyperbolic.proj(res, c=c)
        # head[..., 1] = head[..., 1].clamp(max=-0.1)
        # tail[..., 1] = tail[..., 1].clamp(max=-0.1)

        # margin = -1e-5
        # # pdb.set_trace()
        # head[..., 1] = torch.where(head[..., 1] < margin, head[..., 1], torch.ones_like(head[..., 1]) * margin)
        # tail[..., 1] = torch.where(tail[..., 1] < margin, tail[..., 1], torch.ones_like(tail[..., 1]) * margin)

        MODE = 1 # 1

        if MODE == 0:
            head = torch.stack(torch.chunk(head, 2, dim=2), dim=3)
            tail = torch.stack(torch.chunk(tail, 2, dim=2), dim=3)
            score = hyperbolic.sqdist(head, tail, c)
            score = (score.squeeze(3) + 1e-4) ** 0.5
            score = score.mean(dim=-1)
        else:
            score = hyperbolic.sqdist(head, tail, c)
            score = (score.squeeze(2) + 1e-4) ** 0.5

        return score

    def RotatCones(self, head, relation, tail, mode):
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = True

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            raise ValueError
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)
            res = hyperbolic.proj(res, c=c)
            res = hyperbolic.mobius_add(res, translation, c)
            res = hyperbolic.proj(res, c=c)

            energy_radius = bh + bt - hyperbolic.sqdist(res, tail, c)

            res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            tail = torch.stack([re_tail, im_tail], dim=3)

            tail = self.clip_vector(tail)
            res = self.clip_vector(res)

            energy_1 = self.score_cones(res, tail)
            energy_2 = self.score_cones(tail, res)

        # one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        batch_size = relation_category.shape[0]
        one_many_mask = (relation_category == 1).view(batch_size, 1, 1)
        many_one_mask = (relation_category == 2).view(batch_size, 1, 1)

        score_radius = energy_radius
        score_cones = one_many_mask * energy_1 + many_one_mask * energy_2
        return score_radius, score_cones

    def RotatCones2(self, head, relation, tail, mode):
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = True

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            # raise ValueError
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if translation_flag:
                res = hyperbolic.proj(res,c)
                res = hyperbolic.mobius_add(res, translation, c)

            hyperbolic.proj(res, c)
            energy_0 = 6.0 - hyperbolic.sqdist(res, tail, c)

            res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            tail = torch.stack([re_tail, im_tail], dim=3)

            tail = self.clip_vector(tail)
            res = self.clip_vector(res)

            energy_1 = self.score_cones(res, tail)
            energy_2 = self.score_cones(tail, res)


        one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        one_many_mask = (relation_category == 1).unsqueeze(1).unsqueeze(2)
        many_one_mask = (relation_category == 2).unsqueeze(1).unsqueeze(2)

        score_1 = one_one_mask * energy_0
        # score_1 = energy_0
        score_2 = one_many_mask * energy_1 + many_one_mask * energy_2
        score_2 = bh + bt - score_2

        return (score_1, score_2)


    def EmbedCones(self, head, relation, tail, mode):
        # if isinstance(relation, tuple):
        #     relation_category = relation[1]
        #     relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = True

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)
        #
        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation, weight = relation.split([int((self.entity_dim -1 )/2.), self.entity_dim - 1, 1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            # raise ValueError
            phase_relation = relation / (self.embedding_range.item() / pi)
        # phase_relation = phase_relation / phase_relation * pi
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
        else:
            pass
            # Remove Rotation first for experiment
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            res = torch.cat([re_score, im_score], dim=2)

            # res = head
            #
            # if translation_flag:
            #     res = hyperbolic.proj(res,c)
            #     res = hyperbolic.mobius_add(res, translation, c)
            #
            # hyperbolic.proj(res, c)
            # energy_ball = 0.2 - hyperbolic.sqdist(res, tail, c)
            #
            # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            # tail = torch.stack([re_tail, im_tail], dim=3)
            #
            # tail = self.clip_vector(tail)
            # res = self.clip_vector(res)
            #
            # energy_cones = self.score_cones(res, tail)

        # adjust translation to the local frame of reference

        # x_axis = torch.zeros_like(translation)
        # x_axis[..., 0] = 1.
        #
        # translation = x_axis

        # theta = self.compute_rot_angle(head[..., 0], head[..., 1])
        # re_theta = theta.cos()
        # im_theta = theta.sin()
        # re_translation = translation[..., 0]
        # im_translation = translation[..., 1]
        #
        # re_rot_translation = re_translation * re_theta - im_translation * im_theta
        # im_rot_translation = re_translation * im_theta + im_translation * re_theta
        # rot_translation = torch.stack([re_rot_translation, im_rot_translation], dim=2)
        #
        # # pdb.set_trace()
        #
        # # # FIXME
        # # dummy_mask = (relation_category == -1).unsqueeze(1).unsqueeze(1)
        # # rot_translation = rot_translation * ~dummy_mask
        # #
        # res = hyperbolic.mobius_add(head, rot_translation, c)
        # res = hyperbolic.proj(res, c=c)
        # energy_ball = 0.0005 - hyperbolic.sqdist(res, tail, c)
        # # energy_ball = energy_ball * ~dummy_mask

        # head = torch.stack([re_head, im_head], dim=3)
        # pdb.set_trace()
        res = head # hyperbolic.mobius_add(res, translation, c)
        energy_ball = -F.logsigmoid(1. - hyperbolic.sqdist(res, tail, c))
        res = head
        res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
        tail = torch.stack([re_tail, im_tail], dim=3)
        energy_cones_1 = self.score_cones(res, tail)
        energy_cones_2 = self.score_cones(tail, res)
        # energy_ball = torch.zeros_like(energy_cones_1)

        # weight = F.softmax(weight, dim=-1) # self.sigmoid(weight)

        weight = self.sigmoid(weight)

        # FIXME: HACK TO RECOGNIZE NEGATIVE SAMPLE
        # pdb.set_trace()

        if tail.shape[1] > 1:
            # print('negative')
            energy_cones = energy_cones_1
            # energy_cones = weight[..., 0].unsqueeze(2) * energy_cones_1 + \
            #                weight[..., 1].unsqueeze(2) * energy_cones_2 + \
            #                weight[..., 2].unsqueeze(2) * energy_ball  # 0.5 * (energy_cones_1 + energy_cones_2)

        else:
            # pdb.set_trace()
            # energy_cones = weight[..., 0].unsqueeze(2) * energy_cones_1 + \
            #                weight[..., 1].unsqueeze(2) * energy_cones_2 + \
            #                weight[..., 2].unsqueeze(2) * 0.5 * ((gamma - energy_cones_1).clamp(min=0.) + (gamma - energy_cones_2).clamp(min=0.))
            energy_cones = weight * energy_cones_1 +  (1 - weight) * energy_cones_2


        return energy_ball, energy_cones


    def ConeLCA(self, head, relation, tail, mode):
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = True

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)
        #
        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation, weight = relation.split(
                [int((self.entity_dim - 1) / 2.), self.entity_dim - 1, 1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            # raise ValueError
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
        else:
            # pass
            # Remove Rotation first for experiment
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            res = torch.cat([re_score, im_score], dim=2)

            # res = head
            #
            # if translation_flag:
            #     res = hyperbolic.proj(res,c)
            #     res = hyperbolic.mobius_add(res, translation, c)
            #
            # hyperbolic.proj(res, c)
            # energy_ball = 0.2 - hyperbolic.sqdist(res, tail, c)
            #
            # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            # tail = torch.stack([re_tail, im_tail], dim=3)
            #
            # tail = self.clip_vector(tail)
            # res = self.clip_vector(res)
            #
            # energy_cones = self.score_cones(res, tail)

        # adjust translation to the local frame of reference

        # x_axis = torch.zeros_like(translation)
        # x_axis[..., 0] = 1.
        #
        # translation = x_axis

        # theta = self.compute_rot_angle(head[..., 0], head[..., 1])
        # re_theta = theta.cos()
        # im_theta = theta.sin()
        # re_translation = translation[..., 0]
        # im_translation = translation[..., 1]
        #
        # re_rot_translation = re_translation * re_theta - im_translation * im_theta
        # im_rot_translation = re_translation * im_theta + im_translation * re_theta
        # rot_translation = torch.stack([re_rot_translation, im_rot_translation], dim=2)
        #
        # # pdb.set_trace()
        #
        # # # FIXME
        # # dummy_mask = (relation_category == -1).unsqueeze(1).unsqueeze(1)
        # # rot_translation = rot_translation * ~dummy_mask
        # #
        # res = hyperbolic.mobius_add(head, rot_translation, c)
        # res = hyperbolic.proj(res, c=c)
        # energy_ball = 0.0005 - hyperbolic.sqdist(res, tail, c)
        # # energy_ball = energy_ball * ~dummy_mask

        # head = torch.stack([re_head, im_head], dim=3)
        # pdb.set_trace()


        # # # # # # # #
        #    LCA      #
        # # # # # # # #

        # res = head  # hyperbolic.mobius_add(res, translation, c)
        # energy_ball = 6.0 - 20. * hyperbolic.sqdist(res, tail, c)

        # energy_ball = None

        # res = head
        # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
        # tail = torch.stack([re_tail, im_tail], dim=3)

        # pdb.set_trace()

        energy_ball = 12.0 - 10. * hyperbolic.sqdist(res, tail, c)

        # one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        one_many_mask = (relation_category == 1).unsqueeze(1).unsqueeze(2)
        many_one_mask = (relation_category == 2).unsqueeze(1).unsqueeze(2)

        res = head
        # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
        # tail = torch.stack([re_tail, im_tail], dim=3)

        energy_cones_1 = self.score_cones(res, tail).unsqueeze(2)
        energy_cones_2 = self.score_cones(tail, res).unsqueeze(2)

        # energy_ball = energy_ball * one_one_mask
        energy_cones = energy_cones_1 * one_many_mask + energy_cones_2 * many_one_mask

        return energy_ball, energy_cones


    def BoxLCA(self, head, relation, tail, mode):
        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = False

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head_center, head_offset = torch.chunk(head, 2, dim=2)
        tail_center, tail_offset = torch.chunk(tail, 2, dim=2)

        head_center = hyperbolic.proj(hyperbolic.expmap0(head_center, c), c=c)
        tail_center = hyperbolic.proj(hyperbolic.expmap0(tail_center, c), c=c)
        head_offset = hyperbolic.proj(hyperbolic.expmap0(head_offset, c), c=c)
        tail_offset = hyperbolic.proj(hyperbolic.expmap0(tail_offset, c), c=c)

        # re_head, im_head = torch.chunk(head, 2, dim=2)
        # re_tail, im_tail = torch.chunk(tail, 2, dim=2)
        #
        # # Make phases of relations uniformly distributed in [-pi, pi]
        # if translation_flag:
        #     phase_relation, translation, weight = relation.split(
        #         [int((self.entity_dim - 1) / 2.), self.entity_dim - 1, 1], dim=2)
        #     phase_relation = phase_relation / (self.embedding_range.item() / pi)
        #     translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        # else:
        #     # raise ValueError
        #     phase_relation = relation / (self.embedding_range.item() / pi)
        # # phase_relation = phase_relation / phase_relation * pi
        # re_relation = torch.cos(phase_relation)
        # im_relation = torch.sin(phase_relation)

        # if mode == 'head-batch':
        #     raise ValueError
        # else:
        #     pass
        #     # # Remove Rotation first for experiment
        #     # re_score = re_head * re_relation - im_head * im_relation
        #     # im_score = re_head * im_relation + im_head * re_relation
        #     # res = torch.cat([re_score, im_score], dim=2)
        #
        #     # res = head
        #     #
        #     # if translation_flag:
        #     #     res = hyperbolic.proj(res,c)
        #     #     res = hyperbolic.mobius_add(res, translation, c)
        #     #
        #     # hyperbolic.proj(res, c)
        #     # energy_ball = 0.2 - hyperbolic.sqdist(res, tail, c)
        #     #
        #     # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
        #     # tail = torch.stack([re_tail, im_tail], dim=3)
        #     #
        #     # tail = self.clip_vector(tail)
        #     # res = self.clip_vector(res)
        #     #
        #     # energy_cones = self.score_cones(res, tail)

        # adjust translation to the local frame of reference

        # x_axis = torch.zeros_like(translation)
        # x_axis[..., 0] = 1.
        #
        # translation = x_axis

        # theta = self.compute_rot_angle(head[..., 0], head[..., 1])
        # re_theta = theta.cos()
        # im_theta = theta.sin()
        # re_translation = translation[..., 0]
        # im_translation = translation[..., 1]
        #
        # re_rot_translation = re_translation * re_theta - im_translation * im_theta
        # im_rot_translation = re_translation * im_theta + im_translation * re_theta
        # rot_translation = torch.stack([re_rot_translation, im_rot_translation], dim=2)
        #
        # # pdb.set_trace()
        #
        # # # FIXME
        # # dummy_mask = (relation_category == -1).unsqueeze(1).unsqueeze(1)
        # # rot_translation = rot_translation * ~dummy_mask
        # #
        # res = hyperbolic.mobius_add(head, rot_translation, c)
        # res = hyperbolic.proj(res, c=c)
        # energy_ball = 0.0005 - hyperbolic.sqdist(res, tail, c)
        # # energy_ball = energy_ball * ~dummy_mask

        # head = torch.stack([re_head, im_head], dim=3)
        # pdb.set_trace()


        # # # # # # # #
        #    LCA      #
        # # # # # # # #

        # res = head  # hyperbolic.mobius_add(res, translation, c)
        # energy_ball = 6.0 - 20. * hyperbolic.sqdist(res, tail, c)

        energy_ball = None

        # res = head
        # res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
        # tail = torch.stack([re_tail, im_tail], dim=3)

        # one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        one_many_mask = (relation_category == 1).unsqueeze(1).unsqueeze(2)
        many_one_mask = (relation_category == 2).unsqueeze(1).unsqueeze(2)

        energy_boxes_1 = self.score_boxes(head_center, head_offset, tail_center).unsqueeze(2)
        energy_boxes_2 = self.score_boxes(tail_center, tail_offset, head_center).unsqueeze(2)

        # energy_ball = energy_ball * one_one_mask
        energy_boxes = energy_boxes_1 * one_many_mask + energy_boxes_2 * many_one_mask

        return energy_ball, energy_boxes

    def score_boxes(self, center, offset, vector):
        qmin = center - 0.5 * F.relu(offset)
        qmax = center + 0.5 * F.relu(offset)
        score_offset = F.relu(qmin - vector) + F.relu(vector - qmax)
        # score_center = center - torch.min(qmax, torch.max(qmin, vector))
        # score = torch.norm(score_offset, p=1, dim=-1) + self.center_alpha * torch.norm(score_center, p=1, dim=-1)
        score = torch.norm(score_offset, p=1, dim=-1)
        return score

    def compute_rot_angle(self, x, y):
        pi = 3.14159265358979323846
        x_neg_mask = (x < 0) * (-pi)
        x_zero_mask = x == 0
        y_zero_mask = y == 0

        zero_mask = x_zero_mask & y_zero_mask
        if zero_mask.any():
            raise ValueError('polar angle not defined at origin')
        if x_zero_mask.any():
            print('zero x encounteres')
            x = x + x_zero_mask * 1e-5

        theta = ((y * ~x_zero_mask) / (x * ~x_zero_mask)).atan() + x_neg_mask
        return theta

    def RotatCones3(self, head, relation, tail, mode):

        if isinstance(relation, tuple):
            relation_category = relation[1]
            relation = relation[0]

        pi = 3.14159265358979323846
        translation_flag = True

        # project to hyperbolic manifold
        c = self.softplus(self.curvature)

        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:

            phase_relation, scaling = relation.split([int((self.entity_dim - 1) / 2.), int((self.entity_dim - 1) / 2.)], dim=2)
            scaling = self.softplus(scaling)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
        else:
            # raise ValueError
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            raise ValueError
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if translation_flag:
                res = hyperbolic.logmap0(res, c)
                re, im = torch.chunk(res, 2, dim=2)
                re *= scaling
                im *= scaling
                res = torch.cat([re, im], dim=2)
                res = hyperbolic.expmap0(res, c)

            hyperbolic.proj(res, c)
            energy_0 = 6.0 - hyperbolic.sqdist(res, tail, c)

            res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            tail = torch.stack([re_tail, im_tail], dim=3)

            tail = self.clip_vector(tail)
            res = self.clip_vector(res)

            energy_1 = self.score_cones(res, tail)
            energy_2 = self.score_cones(tail, res)

        one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        one_many_mask = (relation_category == 1).unsqueeze(1).unsqueeze(2)
        many_one_mask = (relation_category == 2).unsqueeze(1).unsqueeze(2)

        score_1 = one_one_mask * energy_0
        # score_1 = energy_0
        score_2 = one_many_mask * energy_1 + many_one_mask * energy_2

        return (score_1, score_2)

    def score_cones(self, x, y):
        energy = self.manifold.angle_at_u(x, y) - self.manifold.half_aperture(x)
        return energy


    def clip_vector(self, embedding):
        lower_bound = self.inner_radius + self.EPS
        upper_bound = 1.0 - self.EPS
        norms = embedding.norm(dim=-1, keepdim=True)
        embedding = torch.where(norms < lower_bound, embedding * (lower_bound / norms), embedding)
        embedding = torch.where(norms > upper_bound, embedding * (upper_bound / norms), embedding)
        return embedding


    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step, viable_neg):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()


        train_iterator.dataloader.dataset.step = step

        if args.train_with_degree:
            positive_sample, negative_sample, subsampling_weight, mode, degree = next(train_iterator)
        else:
            degree = None
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            if isinstance(negative_sample, list):
                negative_sample = [negative_sample[0].cuda(), negative_sample[1].cuda()]
            else:
                negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            degree = None if degree is None else degree.cuda()

        # if args.gamma > 1.0:
        #     print('warning: setting the right gamma')
        #     pdb.set_trace()

        if args.model in ['RotatCones','RotatCones3']:

            negative_energy = model((positive_sample, negative_sample), mode=mode)

            if isinstance(negative_energy, tuple):
                negative_energy_1, negative_energy_2 = negative_energy

                negative_score_1 = negative_energy_1.squeeze(dim=2)

                if args.negative_adversarial_sampling:
                    # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                    negative_score_1 = - (F.softmax(negative_score_1 * args.adversarial_temperature, dim=1).detach()
                                      * F.logsigmoid(-negative_score_1)).sum(dim=1)
                else:
                    negative_score_1 = - F.logsigmoid(-negative_score_1).mean(dim=1)

                negative_score_2 = (args.gamma - negative_energy_2.clamp(min=0)).clamp(min=0).mean(dim=2)
                # negative_score_2 = (F.softmax(negative_score_2 * args.adversarial_temperature, dim=2).detach()
                #                     * negative_score_2).sum(dim=2)

                if args.negative_adversarial_sampling:
                    negative_score_2 = (F.softmax(negative_score_2 * args.adversarial_temperature, dim=1).detach()
                                        * negative_score_2).sum(dim=1)
                else:
                    negative_score_2 = negative_score_2.mean(dim=1)

                one_one_mask = (positive_sample[:, 3] == 0)
                if args.model == 'RotatCones': # consider radius for all relations
                    negative_score = negative_score_1 + ~one_one_mask * negative_score_2
                else:
                    # raise ValueError
                    negative_score = one_one_mask * negative_score_1 + ~one_one_mask * negative_score_2
            else:
                negative_score = (args.gamma - negative_energy).clamp(min=0).mean(dim=2)
                negative_score = negative_score.sum(dim=1)

            positive_energy = model(positive_sample)

            if isinstance(positive_energy, tuple):
                positive_energy_1, positive_energy_2 = positive_energy
                positive_score_1 = - F.logsigmoid(positive_energy_1.squeeze(dim=2)).squeeze(dim = 1)
                positive_score_2 = positive_energy_2.clamp(min=0).mean(dim=2).squeeze(dim=1)
                # positive_score_2 = (F.softmax(positive_score_2 * args.adversarial_temperature, dim=2).detach()
                #                     * positive_score_2).sum(dim=2).squeeze(dim=1)

                if args.model == 'RotatCones':
                    positive_score = positive_score_1 + ~one_one_mask * positive_score_2
                else:
                    # raise ValueError
                    positive_score = one_one_mask * positive_score_1 + ~one_one_mask * positive_score_2

            else:
                positive_score = positive_energy.clamp(min=0).mean(dim=2).squeeze(dim=1)

            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2.0

            if step % 100 == 0:

                n_1 = torch.sum(one_one_mask) + 1e-15
                n_2 = one_one_mask.shape[0] - n_1 + 1e-15
                positive_loss_1 = torch.sum(one_one_mask * positive_score_1) / n_1
                negative_loss_1 = torch.sum(one_one_mask * negative_score_1) / n_1
                positive_loss_2 = torch.sum(~one_one_mask * positive_score_2) / n_2
                negative_loss_2 = torch.sum(~one_one_mask * negative_score_2) / n_2
                print(positive_loss_1.item(), positive_loss_2.item(), negative_loss_1.item(), negative_loss_2.item())

            loss.backward()
            regularization_log = {}

        elif args.model == 'RotatCones2':
            negative_energy = model((positive_sample, negative_sample), mode=mode)

            assert isinstance(negative_energy, tuple)
            negative_energy_1, negative_energy_2 = negative_energy
            negative_score_1 = negative_energy_1.squeeze(dim=2)

            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score_1 = - (F.softmax(negative_score_1 * args.adversarial_temperature, dim=1).detach()
                                      * F.logsigmoid(-negative_score_1)).sum(dim=1)
            else:
                negative_score_1 = - F.logsigmoid(-negative_score_1).mean(dim=1)

            negative_score_2 = F.logsigmoid(-negative_energy_2)# .mean(dim=2).mean(dim=1)
            negative_score_2 = (F.softmax(negative_score_2 * args.adversarial_temperature, dim=2).detach() * negative_score_2).sum(dim=2)
            negative_score_2 = (F.softmax(negative_score_2 * args.adversarial_temperature,
                                          dim=1).detach() * negative_score_2).sum(dim=1)
            negative_score_2 = - negative_score_2

            # if args.negative_adversarial_sampling:
            #     negative_score_2 = - (F.softmax(negative_score_2 * args.adversarial_temperature, dim=1).detach()
            #                         * F.logsigmoid(-negative_score_2)).sum(dim=1)
            # else:
            #     negative_score_2 = negative_score_2.mean(dim=1)

            one_one_mask = (positive_sample[:, 3] == 0)
            negative_score = one_one_mask * negative_score_1 + ~one_one_mask * negative_score_2

            positive_energy = model(positive_sample)

            assert isinstance(positive_energy, tuple)
            positive_energy_1, positive_energy_2 = positive_energy
            positive_score_1 = - F.logsigmoid(positive_energy_1.squeeze(dim=2)).squeeze(dim=1)
            positive_score_2 = - F.logsigmoid(positive_energy_2).mean(dim=2).squeeze(dim=1)
            # positive_score_2 = positive_energy_2.clamp(min=0).mean(dim=2).squeeze(dim=1)

            positive_score = one_one_mask * positive_score_1 + ~one_one_mask * positive_score_2
            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2.0

            if step % 10 == 0:
                n_1 = torch.sum(one_one_mask) + 1e-15
                n_2 = one_one_mask.shape[0] - n_1 + 1e-15
                positive_loss_1 = torch.sum(one_one_mask * positive_score_1) / n_1
                negative_loss_1 = torch.sum(one_one_mask * negative_score_1) / n_1
                positive_loss_2 = torch.sum(~one_one_mask * positive_score_2) / n_2
                negative_loss_2 = torch.sum(~one_one_mask * negative_score_2) / n_2
                print(positive_loss_1.item(), positive_loss_2.item(), negative_loss_1.item(), negative_loss_2.item())

            loss.backward()
            regularization_log = {}

        elif args.model == 'PoincareEmbedding':

            # # # negative sample sanity check:
            # # non_hierarchical_relations = [7, 8, 11, 13]
            # for i in range(positive_sample.shape[0]):
            #     head = positive_sample[i, 0].item()
            #     relation = positive_sample[i, 1].item()
            #
            #     neg = negative_sample[i, :].tolist()
            #     vneg = viable_neg[str(head)]
            #     for k in vneg:
            #         if k in neg:
            #             pdb.set_trace()


            # # select from two negative samples according to relation
            # non_hierarchical_relations = [7, 8, 11, 13]
            #
            # mask = None
            # for r in non_hierarchical_relations:
            #     if mask is None:
            #         mask = positive_sample[:, 1] == r
            #     else:
            #         mask = mask | (positive_sample[:, 1] == r)
            #
            # mask = mask.unsqueeze(1)
            # negative_sample = negative_sample[0] * ~mask + negative_sample[1] * mask

            negative_score = model((positive_sample, negative_sample), mode=mode)
            negative_score = negative_score
            positive_score = model(positive_sample)
            score = torch.cat([positive_score, negative_score], dim=1)
            loss = F.cross_entropy(score.neg(), torch.zeros(score.shape[0],).long().cuda())
            loss.backward()
            positive_sample_loss = torch.zeros_like(loss)
            negative_sample_loss = torch.zeros_like(loss)
            regularization_log = {}
            if step % 50 == 0:

                plt.figure(figsize=(10, 10))
                embedding = model.entity_embedding.cpu().data
                c = model.softplus(model.curvature.cpu().data)
                # res = res.cpu().data
                all_embedding = hyperbolic.proj(hyperbolic.expmap0(embedding, c), c=c)
                plt.scatter(all_embedding[:, 0], all_embedding[:, 1], s=0.5, color='gray')
                # plt.scatter(res[:, 0,0, 0], res[:, 0,0,1], s=10.0, color='red')

                plt.scatter(0, 0, s=10.0, color='red')
                # n = [i for i in [0, 866, 13, 1623, 2, 677, 3, 200, 4, 1227, 1649, 263, 29, 183, 1146, 888]]
                # for i, txt in enumerate(n):
                #     plt.annotate(txt, (all_embedding[txt, 0], all_embedding[txt, 1]))

                # n = [positive_sample[i, 2].item() for i in range(res.shape[0])]
                # for i, txt in enumerate(n):
                #     plt.annotate(txt, (res[i, 0,0, 0], res[i, 0,0, 1]),color='r')

                radius = all_embedding.abs().max()
                plt.xlim([-radius, radius])
                plt.ylim([-radius, radius])
                plt.savefig('./misc/figures_2/%d.png' % step)
                plt.close()


        elif args.model == 'ConeLCA':

            if isinstance(negative_sample, list):
                mask = (positive_sample[:, 3] == 0).unsqueeze(1)
                negative_sample = negative_sample[0] * ~mask + negative_sample[1] * mask

            model.entity_embedding.data[model.dummy_node, :] = 1e-7

            negative_energy_1, negative_energy_2 = model((positive_sample, negative_sample), mode=mode)

            negative_score_1 = negative_energy_1.squeeze(dim=2)

            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score_1 = - (F.softmax(negative_score_1 * args.adversarial_temperature, dim=1).detach()
                                      * F.logsigmoid(-negative_score_1)).sum(dim=1)
            else:
                negative_score_1 = - F.logsigmoid(-negative_score_1).mean(dim=1)


            negative_score_2 = (args.gamma - negative_energy_2).clamp(min=0).mean(dim=2)
            negative_score_2 = negative_score_2.mean(dim=1)
            # negative_score_2 = negative_score_2.sum(dim=1)

            # Masking
            dummy_node_mask = (positive_sample[:, 2] == model.dummy_node)
            one_one_mask = (positive_sample[:, 3] == 0)

            # precision = model.log_vars.neg().exp()
            # negative_score_1 *= precision[0]
            # negative_score_2 *= precision[1]

            positive_energy_1, positive_energy_2 = model(positive_sample)

            positive_score_1 = - F.logsigmoid(positive_energy_1.squeeze(dim=2)).squeeze(dim=1)
            positive_score_2 = positive_energy_2.clamp(min=0).mean(dim=2).squeeze(dim=1)


            positive_score_1 = positive_score_1 * one_one_mask
            positive_score_2 = positive_score_2 * ~dummy_node_mask # do not include positive score for leaf node case
            negative_score_1 = negative_score_1 * one_one_mask
            positive_score = positive_score_1 + positive_score_2
            negative_score = negative_score_1 + negative_score_2

            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2.0 # + model.log_vars.sum()
            loss.backward()

            if step % 100 == 0:
                print(positive_score_1.mean().item(), positive_score_2.mean().item(), negative_score_1.mean().item(), negative_score_2.mean().item())

            if step % 100 == 0:
                # torch.save({
                #     'cone_pos_loss': positive_score_2.mean().item(),
                #     'cone_neg_loss': negative_score_2.mean().item(),
                #     'ball_pos_loss': positive_score_1.mean().item(),
                #     'ball_neg_loss': negative_score_1.mean().item(),
                #     'total_loss': loss.item() * 2.,
                #     'model_state_dict': model.state_dict()},
                #     os.path.join(args.save_path, 'vis_checkpoint', 'ckpt_%d' % step)
                # )

                plt.figure(figsize=(30, 30))
                embedding = model.entity_embedding.cpu().data
                c = model.softplus(model.curvature.cpu().data)

                K = list()
                for i in viable_neg.keys():
                    if len(viable_neg[i]) > 100:
                        K.append(i)

                N = 3
                rand_indices = np.random.randint(len(K), size=N*N)
                for i in range(N * N):
                    plt.subplot(N, N, i+1)

                    node = K[rand_indices[i]]
                    descendants = viable_neg[node]

                    # res = res.cpu().data
                    all_embedding = hyperbolic.proj(hyperbolic.expmap0(embedding, c), c=c)
                    plt.scatter(all_embedding[:, 0], all_embedding[:, 1], s=0.1, color='gray')
                    plt.scatter(all_embedding[descendants, 0], all_embedding[descendants, 1], s=10.0, color='yellow')
                    plt.scatter(all_embedding[int(node), 0], all_embedding[int(node), 1], s=10.0, color='blue')


                    # plt.scatter(res[:, 0,0, 0], res[:, 0,0,1], s=10.0, color='red')

                    plt.scatter(0, 0, s=10.0, color='red')

                    radius = all_embedding.abs().max() + 0.05
                    plt.xlim([-radius, radius])
                    plt.ylim([-radius, radius])
                plt.savefig('./misc/figures/%d.png' % step)
                plt.close()

            regularization_log = {}

        elif args.model == 'BoxLCA':

            if isinstance(negative_sample, list):
                mask = (positive_sample[:, 3] == 0).unsqueeze(1)
                negative_sample = negative_sample[0] * ~mask + negative_sample[1] * mask

            model.entity_embedding.data[model.dummy_node, :] = 1e-7

            negative_energy_1, negative_energy_2 = model((positive_sample, negative_sample), mode=mode)

            # negative_score_1 = negative_energy_1.squeeze(dim=2)
            #
            # if args.negative_adversarial_sampling:
            #     # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            #     negative_score_1 = - (F.softmax(negative_score_1 * args.adversarial_temperature, dim=1).detach()
            #                           * F.logsigmoid(-negative_score_1)).sum(dim=1)
            # else:
            #     negative_score_1 = - F.logsigmoid(-negative_score_1).mean(dim=1)

            # pdb.set_trace()
            negative_score_2 = (args.gamma - negative_energy_2).clamp(min=0).mean(dim=2)
            negative_score_2 = negative_score_2.mean(dim=1)
            # negative_score_2 = negative_score_2.sum(dim=1)

            # Masking
            dummy_node_mask = (positive_sample[:, 2] == model.dummy_node)
            one_one_mask = (positive_sample[:, 3] == 0)

            # precision = model.log_vars.neg().exp()
            # negative_score_1 *= precision[0]
            # negative_score_2 *= precision[1]

            positive_energy_1, positive_energy_2 = model(positive_sample)

            # positive_score_1 = - F.logsigmoid(positive_energy_1.squeeze(dim=2)).squeeze(dim=1)
            positive_score_2 = positive_energy_2.mean(dim=2).squeeze(dim=1)


            # positive_score_1 = positive_score_1 * one_one_mask
            positive_score_2 = positive_score_2 * ~dummy_node_mask # do not include positive score for leaf node case
            # negative_score_1 = negative_score_1 * one_one_mask
            positive_score = positive_score_2
            negative_score = negative_score_2

            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2.0 # + model.log_vars.sum()
            loss.backward()

            # if step % 100 == 0:
                # print(positive_score_1.mean().item(), positive_score_2.mean().item(), negative_score_1.mean().item(), negative_score_2.mean().item())

            if step % 100 == 0:
                # torch.save({
                #     'cone_pos_loss': positive_score_2.mean().item(),
                #     'cone_neg_loss': negative_score_2.mean().item(),
                #     'ball_pos_loss': positive_score_1.mean().item(),
                #     'ball_neg_loss': negative_score_1.mean().item(),
                #     'total_loss': loss.item() * 2.,
                #     'model_state_dict': model.state_dict()},
                #     os.path.join(args.save_path, 'vis_checkpoint', 'ckpt_%d' % step)
                # )

                plt.figure(figsize=(30, 30))
                embedding = model.entity_embedding.cpu().data
                c = model.softplus(model.curvature.cpu().data)

                K = list()
                for i in viable_neg.keys():
                    if len(viable_neg[i]) > 100:
                        K.append(i)

                N = 3
                rand_indices = np.random.randint(len(K), size=N*N)
                for i in range(N * N):
                    plt.subplot(N, N, i+1)

                    node = K[rand_indices[i]]
                    descendants = viable_neg[node]

                    # res = res.cpu().data
                    all_embedding = hyperbolic.proj(hyperbolic.expmap0(embedding, c), c=c)
                    plt.scatter(all_embedding[:, 0], all_embedding[:, 1], s=0.1, color='gray')
                    plt.scatter(all_embedding[descendants, 0], all_embedding[descendants, 1], s=10.0, color='yellow')
                    plt.scatter(all_embedding[int(node), 0], all_embedding[int(node), 1], s=10.0, color='blue')


                    # plt.scatter(res[:, 0,0, 0], res[:, 0,0,1], s=10.0, color='red')

                    plt.scatter(0, 0, s=10.0, color='red')

                    radius = all_embedding.abs().max() + 0.05
                    plt.xlim([-radius, radius])
                    plt.ylim([-radius, radius])
                plt.savefig('./misc/figures/%d.png' % step)
                plt.close()

            regularization_log = {}



        elif args.model == 'EmbedCones':

            hold_out = [0, 1, 3, 7, 12, 15, 21, 25, 46, 76, 77, 88, 104]
            five_percent = [(0, 1), (1, 1), (3, 1), (7, 1), (21, 0), (46, 1), (76, 1)]
            ten_percent = [(0, 1), (1, 1), (3, 1), (7, 1), (12, 2), (15, 0), (21, 0), (25, 1), (46, 1), (76, 1), (77, 1), (88, 1), (104, 1)]
            semi_supervised = None


            # model.entity_embedding.data[0, :] = model.root
            model.entity_embedding.data[-1, :] = 1e-7

            # print(positive_sample)
            negative_energy = model((positive_sample, negative_sample), mode=mode)

            negative_energy_1, negative_energy_2 = negative_energy
            negative_score_1 = negative_energy_1.squeeze(dim=2)

            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score_1 = - (F.softmax(negative_score_1 * args.adversarial_temperature, dim=1).detach()
                                      * F.logsigmoid(-negative_score_1)).sum(dim=1)
            else:
                negative_score_1 = - F.logsigmoid(-negative_score_1).mean(dim=1)


            negative_score_2 = (args.gamma - negative_energy_2).clamp(min=0).mean(dim=2)

            # for i in range(negative_sample.shape[0]):
            #     head = positive_sample[i, 0].item()
            #     vn = viable_neg[head]
            #     if not head == 0:
            #         for j in range(negative_sample.shape[1]):
            #             if not negative_sample[i, j].item() in vn:
            #                 print(negative_sample[i, j].item(), head, vn)
            #                 assert False


            # if step % 100 == 0:
            #     print(step)
            #     for i in range(negative_score_2.shape[0]):
            #         print(positive_sample[i, ...].tolist(), negative_sample[i, :].tolist(), negative_score_2[i, :].tolist())
            #     print('-' * 10)

            # negative_score_2 = negative_score_2.mean(dim=1)
            negative_score_2 = negative_score_2.sum(dim=1)


            # masking
            if torch.abs(torch.randn(1, )) < 0.01: print('Warning: root mask and dummy relation mask')
            # root_mask = (positive_sample[:, 0] == 0)
            # dummy_relation_mask = (positive_sample[:, 3] == -1)

            dummy_node_mask = (positive_sample[:, 2] == model.dummy_node)
            # negative_score_1 = negative_score_1 * (~dummy_relation_mask & ~root_mask)
            # negative_score_2 = negative_score_2 * ~root_mask

            # precision = model.log_vars.neg().exp()
            # negative_score_1 *= precision[0]
            # negative_score_2 *= precision[1]
            # 
            # negative_score = negative_score_1 + negative_score_2
            negative_score_2 = torch.zeros_like(negative_score_2)
            negative_score = negative_score_2

            # pdb.set_trace()

            positive_energy = model(positive_sample)


            assert isinstance(positive_energy, tuple)
            positive_energy_1, positive_energy_2 = positive_energy

            positive_score_1 = - F.logsigmoid(positive_energy_1.squeeze(dim=2)).squeeze(dim=1)
            positive_score_2 = positive_energy_2.clamp(min=0).mean(dim=2).squeeze(dim=1)

            # positive_score_1 *= precision[0]
            # positive_score_2 *= precision[1]
            # positive_score = positive_score_1 + positive_score_2

            positive_score_2 = positive_score_2 * ~dummy_node_mask
            positive_score = positive_score_2

            if args.uni_weight:
                positive_sample_loss = positive_score.mean()
                negative_sample_loss = negative_score.mean()
            else:
                positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2.0 # + model.log_vars.sum()

            if semi_supervised is not None:
                weight_list = []
                gt_list = []
                if semi_supervised == 'five_percent':
                    for index, gt in five_percent:
                        weight_list.append(model.sigmoid(model.relation_embedding[index, -1]).item())
                        gt_list.append(gt)
                # pdb.set_trace()
                weight = torch.Tensor(weight_list)
                gt = torch.Tensor(gt_list)
                gt = (gt == 0).float()
                bce_loss = nn.BCELoss()(weight, gt)
                # print(weight, gt)
                loss = loss + bce_loss
            else:
                bce_loss = None

            loss.backward()

            if step % 100 == 0:
                # hold-out set for weakly supervised:
                if semi_supervised is not None:
                    print('semi_supervised', semi_supervised)
                relation2gt = {'works.for': [0, 1], 'proxy.for': [1, 2], 'agent.competes.with.agent': [2, 2], 'subpart.of': [3, 1], 'location.located.within.location': [7, 1], 'company.economic.sector': [9, 1], 'agent.belongs.to.organization': [11, 1], 'agent.collaborates.with.agent': [12, 2], 'top.member.of.organization': [14, 1], 'organization.hired.person': [15, 0], 'object.found.in.scene': [16, 1], 'body.part.contains.body.part': [19, 0], 'object.part.of.object': [20, 1], 'agent.created': [21, 0], 'attraction.of.city': [22, 1], 'person.born.in.city': [23, 1], 'person.born.in.location': [24, 1], 'agent.acts.in.location': [25, 1], 'parent.of.person': [26, 0], 'person.belongs.to.organization': [27, 1], 'person.graduated.from.university': [28, 1], 'person.graduated.school': [29, 1], 'has.spouse': [31, 2], 'has.wife': [32, 2], 'tourist.attraction.such.as.tourist.attraction': [34, 2], 'animal.is.type.of.animal': [36, 1], 'specialization.of': [38, 1], 'animal.such.as.fish': [39, 2], 'organization.headquartered.in.city': [40, 1], 'synonym.for': [42, 2], 'building.located.in.city': [43, 1], 'organization.headquartered.in.country': [44, 1], 'subpart.of.organization': [46, 1], 'company.also.known.as': [51, 2], 'competes.with': [52, 2], 'has.husband': [58, 2], 'city.located.in.state': [59, 1], 'organization.headquartered.in.state.or.province': [63, 1], 'person.also.known.as': [65, 2], 'profession.is.type.of.profession': [66, 1], 'politician.represents.location': [68, 1], 'organization.also.known.as': [69, 2], 'coaches.in.league': [73, 1], 'athlete.plays.sport': [76, 1], 'athlete.plays.for.team': [77, 1], 'athlete.plays.in.league': [78, 1], 'has.sibling': [79, 2], 'animal.such.as.invertebrate': [81, 2], 'person.died.in.country': [85, 1], 'transportation.in.city': [86, 1], 'team.plays.in.city': [87, 1], 'team.plays.in.league': [88, 1], 'team.plays.sport': [89, 1], 'athlete.home.stadium': [90, 1], 'athlete.led.sport.steam': [91, 1], 'athlete.plays.sports.team.position': [92, 1], 'person.has.residence.in.geopolitical.location': [94, 1], 'country.also.known.as': [95, 2], 'animal.such.as.insect': [96, 2], 'arthropod.and.other.arthropod': [97, 2], 'arthropod.called.arthropod': [98, 2], 'produces.product': [102, 0], 'city.lies.on.river': [104, 1], 'city.located.in.country': [105, 1], 'city.located.in.geopolitical.location': [106, 1], 'airport.in.city': [108, 1], 'team.also.known.as': [113, 2], 'agricultural.product.cooked.with.agricultural.product': [114, 2], 'lake.in.state': [115, 1], 'mother.of.person': [116, 0], 'team.plays.against.team': [117, 2], 'stadium.located.in.city': [118, 1], 'clothing.to.go.with.clothing': [122, 2], 'hotel.in.city': [127, 1], 'music.genres.such.as.music.genres': [129, 2], 'city.also.known.as': [131, 2], 'television.station.in.city': [133, 1], 'park.in.city': [134, 1], 'state.has.capital': [135, 0], 'state.located.in.country': [136, 1], 'state.located.in.geopolitical.location': [137, 1], 'state.or.province.is.bordered.by.state.or.province': [138, 2], 'museum.in.city': [139, 1], 'city.capital.of.country': [140, 1], 'country.located.in.geopolitical.location': [141, 1], 'country.currency': [142, 0], 'mountain.in.state': [143, 1], 'radio.station.in.city': [144, 1], 'sport.school.in.country': [146, 1], 'sport.fans.in.country': [148, 1], 'hobbies.such.as.hobbies': [150, 0], 'academic.program.at.university': [154, 1], 'academic.field.concerns.subject': [155, 0], 'academic.field.such.as.academic.field': [156, 0], 'language.of.country': [157, 1], 'language.of.university': [158, 0], 'sports.game.team': [161, 0], 'music.artist.genre': [166, 1], 'plant.growing.in.plant': [168, 1], 'mammal.such.as.mammal': [170, 2], 'agricultural.product.including.agricultural.product': [171, 0], 'agricultural.product.growing.in.state.or.province': [178, 1], 'agricultural.product.to.attract.insect': [179, 2], 'artery.called.artery': [180, 2], 'weapon.made.in.country': [182, 1], 'baked.good.served.with.beverage': [183, 2], 'newspaper.in.city': [184, 1], 'politician.us.endorsed.by.politician.us': [185, 2], 'politician.us.holds.office': [186, 1], 'politician.us.member.of.political.group': [187, 1], 'father.of.person': [188, 0], 'direct.or.directed.movie': [189, 0], 'visual.artist.art.form': [190, 1], 'athlete.beat.athlete': [191, 2], 'automobile.maker.dealers.in.country': [192, 1], 'automaker.produces.model': [193, 0], 'automobile.maker.car.dealers.in.state.or.province': [194, 1], 'automobile.maker.dealers.in.city': [195, 1], 'bank.bank.in.country': [196, 1], 'league.stadiums': [197, 0]}
                ntest = len(relation2gt.keys()) - len(hold_out)
                # Compute classification accuracy
                acc = []
                for thresh in range(300):
                    threshold = thresh * 0.001
                    count = 0
                    # pdb.set_trace()
                    for key, value in relation2gt.items():
                        rid, gt = value
                        if rid not in hold_out:
                            weight = model.relation_embedding[rid, -1]
                            weight = model.sigmoid(weight)
                            if weight >= (0.5 + threshold):
                                pred = 0
                            elif weight <= (0.5 - threshold):
                                pred = 1
                            else:
                                pred = 2

                            if pred == gt:
                                count += 1


                    acc.append((count / ntest, threshold))
                    # print('accuracy: ', count / len(relation2gt.keys()), threshold)
                max_acc = max(acc, key = itemgetter(0))
                print('hold out accuracy / threshold:', max_acc, ntest)
                threshold = max_acc[1]
                incorrect = []
                print('---------------------')
                for key, value in relation2gt.items():
                    rid, gt = value
                    if rid not in hold_out:
                        weight = model.relation_embedding[rid, -1]
                        weight = model.sigmoid(weight)
                        if weight >= (0.5 + threshold):
                            pred = 0
                        elif weight <= (0.5 - threshold):
                            pred = 1
                        else:
                            pred = 2

                        if not pred == gt:
                            print(rid, key, pred, gt, weight.item())
                            incorrect.append(key)
                # print(incorrect)
                print('---------------------')
                print(positive_score_1.mean().item(), positive_score_2.mean().item(), negative_score_1.mean().item(), negative_score_2.mean().item())
                # indices = torch.argmax(model.relation_embedding[:, -3:], dim=-1)
                # hyponym = []
                # hypernym = []
                # symmetry = []



                #     pred = indices[i].item()
                #     # print(i, r, pred)
                #     if pred == 0:
                #         hyponym.append(r)
                #     elif pred == 1:
                #         hypernym.append(r)
                #     elif pred == 2:
                #         symmetry.append(r)
                # print('------------ Hypernym ------------')
                # print(hypernym)
                # print('------------ Hyponym ------------')
                # print(hyponym)
                # print('------------ Symmetry ------------')
                # print(symmetry)
                # print(indices)

            # if step % 10 == 0:
            #     # torch.save({
            #     #     'cone_pos_loss': positive_score_2.mean().item(),
            #     #     'cone_neg_loss': negative_score_2.mean().item(),
            #     #     'ball_pos_loss': positive_score_1.mean().item(),
            #     #     'ball_neg_loss': negative_score_1.mean().item(),
            #     #     'total_loss': loss.item() * 2.,
            #     #     'model_state_dict': model.state_dict()},
            #     #     os.path.join(args.save_path, 'vis_checkpoint', 'ckpt_%d' % step)
            #     # )
            #
            #     plt.figure(figsize=(10, 10))
            #     embedding = model.entity_embedding.cpu().data
            #     c = model.softplus(model.curvature.cpu().data)
            #     # res = res.cpu().data
            #     all_embedding = hyperbolic.proj(hyperbolic.expmap0(embedding, c), c=c)
            #     plt.scatter(all_embedding[:, 0], all_embedding[:, 1], s=1.0, color='gray')
            #     # plt.scatter(res[:, 0,0, 0], res[:, 0,0,1], s=10.0, color='red')
            #
            #     plt.scatter(0, 0, s=10.0, color='red')
            #     n = [i for i in range(10)]
            #     for i, txt in enumerate(n):
            #         plt.annotate(txt, (all_embedding[i, 0], all_embedding[i, 1]))
            #
            #     # n = [i for i in [0, 333, 5, 641, 6, 90, 278, 255, 11, 38, 365, 556, 16, 72, 442, 342, 20, 90, 278, 439]]
            #     # for i, txt in enumerate(n):
            #     #     plt.annotate(txt, (all_embedding[txt, 0], all_embedding[txt, 1]))
            #
            #     # n = [positive_sample[i, 2].item() for i in range(res.shape[0])]
            #     # for i, txt in enumerate(n):
            #     #     plt.annotate(txt, (res[i, 0,0, 0], res[i, 0,0, 1]),color='r')
            #
            #     radius = all_embedding.abs().max() + 0.05
            #     plt.xlim([-radius, radius])
            #     plt.ylim([-radius, radius])
            #     plt.savefig('./misc/figures/%d.png' % step)
            #     plt.close()


            # regularization_log = {'poincare_loss': poincare_loss.item()}
            # regularization_log = {}
            if bce_loss is None:
                regularization_log = {}
            else:
                regularization_log = {'bce_loss': bce_loss.item()}


        # elif args.model == 'RotatTransH2':
        #     negative_score = model((positive_sample, negative_sample), mode=mode, degree=degree)
        #     negative_score = (args.gamma - negative_score).clamp(min=0)
        #
        #     if args.negative_adversarial_sampling:
        #         # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
        #         negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
        #                           * negative_score).sum(dim=1)
        #     else:
        #         negative_score = negative_score.mean(dim=1)
        #
        #     positive_score = model(positive_sample, degree=degree).squeeze(dim=1)
        #
        #     if args.uni_weight:
        #         positive_sample_loss = positive_score.mean()
        #         negative_sample_loss = negative_score.mean()
        #     else:
        #         positive_sample_loss = (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        #         negative_sample_loss = (subsampling_weight * negative_score).sum() / subsampling_weight.sum()
        #
        #     loss = (positive_sample_loss + negative_sample_loss) / 2
        #
        #     regularization_log = {}
        #     loss.backward()


        else:
            negative_score = model((positive_sample, negative_sample), mode=mode, degree=degree)

            # if torch.abs(torch.randn(1, )) < 0.01:
            #     print('Warning: root mask')
            # root_mask = positive_sample[:, 0] == 0
            # negative_score = negative_score * ~root_mask.unsqueeze(1)

            if args.negative_adversarial_sampling:
                #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim = 1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

            positive_score = model(positive_sample, degree=degree)

            positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss)/2

            if args.regularization != 0.0:
                #Use L3 regularization for ComplEx and DistMult
                regularization = args.regularization * (
                    model.entity_embedding.norm(p = 3)**3 +
                    model.relation_embedding.norm(p = 3).norm(p = 3)**3
                )
                loss = loss + regularization
                regularization_log = {'regularization': regularization.item()}
            else:
                regularization_log = {}

            loss.backward()

        optimizer.step()
        # # Burnin
        # burnin_step = 1000
        # lr_multilier = 0.1
        # if step < burnin_step:
        #     burnin_lr = args.learning_rate * lr_multilier
        #     if step % 1000 == 0:
        #         print('Burnin[%d/%d] lr = %f' % (step, burnin_step, burnin_lr))
        #
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = burnin_lr
        #     optimizer.step()
        # elif step == burnin_step:
        #     print('Change lr back to ', args.learning_rate)
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = args.learning_rate
        #     optimizer.step()
        # else:
        #     optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log


    @staticmethod
    def test_step(model, test_triples, all_true_triples, args, relation_category=False, degree=None):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}

        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'head-batch',
                    degree
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples,
                    all_true_triples,
                    args.nentity,
                    args.nrelation,
                    'tail-batch',
                    degree
                ),
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2),
                collate_fn=TestDataset.collate_fn
            )

            if args.tail_batch_only:
                print('Warning: using tail batch for testing only')
                test_dataset_list = [test_dataloader_tail]
            else:
                test_dataset_list = [test_dataloader_head, test_dataloader_tail]

            logs = []
            hits = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            if relation_category:
                predict_head = {0: None, 1: None, 2: None, 3: None}
                predict_tail = {0: None, 1: None, 2: None, 3: None}

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for data in test_dataset:
                        if args.train_with_degree:
                            positive_sample, negative_sample, filter_bias, mode, degree = data
                        else:
                            degree = None
                            positive_sample, negative_sample, filter_bias, mode = data

                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()
                            degree = None if degree is None else degree.cuda()

                        batch_size = positive_sample.size(0)

                        if args.train_with_relation_category:
                            assert positive_sample.size(1) == 4
                            category = positive_sample[:, 3]
                        else:
                            if relation_category:
                                assert positive_sample.size(1) == 4
                                category = positive_sample[:, 3]
                                positive_sample = positive_sample[:, 0:3]

                        score = model((positive_sample, negative_sample), mode, degree)

                        if isinstance(score, tuple):
                            if args.model == 'RotatCones':
                                score_0 = score[0].squeeze(2)
                                score_1 = score[1].mean(dim=2)
                                score_1 = - score_1

                                one_one_mask = (category == 0).unsqueeze(1)
                                score = score_0 # + ~one_one_mask * score_1
                                del score_0, score_1
                                score += (100 * filter_bias)
                                argsort = torch.argsort(score, dim=1, descending=True)

                            elif args.model in ['RotatCones2', 'RotatCones3']:
                                score_0 = score[0].squeeze(2)
                                score_1 = score[1].mean(dim=2)
                                # score_1 = - score_1

                                one_one_mask = (category == 0).unsqueeze(1)
                                score = one_one_mask * score_0 + ~one_one_mask * score_1
                                del score_0, score_1
                                score += (100 * filter_bias)
                                argsort = torch.argsort(score, dim=1, descending=True)
                            else:
                                raise ValueError

                        else:
                            if args.model in ['RotatCones', 'RotatCones2']:
                                score = score.mean(dim=2)

                            score += (filter_bias * 100)

                            #Explicitly sort all the entities to ensure that there is no test exposure bias
                            if args.model in ['RotatCones', 'RotatCones2']:
                                argsort = torch.argsort(score, dim=1, descending=False)
                            else:
                                argsort = torch.argsort(score, dim = 1, descending=True)

                        if mode == 'head-batch':
                            positive_arg = positive_sample[:, 0]
                        elif mode == 'tail-batch':
                            positive_arg = positive_sample[:, 2]
                        else:
                            raise ValueError('mode %s not supported' % mode)

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            content = {
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            }
                            logs.append(content)

                            # Save top three predictions for analysis


                            if relation_category:
                                if mode == 'head-batch':
                                    if predict_head[category[i].item()] is None:
                                        predict_head[category[i].item()] = []
                                    predict_head[category[i].item()].append(content)
                                elif mode == 'tail-batch':
                                    if predict_tail[category[i].item()] is None:
                                        predict_tail[category[i].item()] = []
                                    predict_tail[category[i].item()].append(content)
                                else:
                                    raise ValueError('mode %s not supported' % mode)

                                hits.append({
                                    'positive': positive_sample[i, :].cpu().data,
                                    'indices': argsort[i, 0:5].cpu().data,
                                    'mode': mode,
                                    'category': category[i].item(),
                                    'ranking': ranking,
                                    'filter_bias': torch.nonzero(filter_bias[i, :]).squeeze(1).cpu().data
                                })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

            if relation_category:
                id2category = {0: '1-1', 1: '1-M', 2:'M-1', 3: 'M-M'}
                for i in range(4):
                    del logs
                    logs = predict_head[i]
                    if logs is not None:
                        for metric in logs[0].keys():
                            prefix = 'predict-head ' + id2category[i] + ' ' + metric
                            metrics[prefix] = sum([log[metric] for log in logs])/len(logs)

                for i in range(4):
                    del logs
                    logs = predict_tail[i]
                    if logs is not None:
                        for metric in logs[0].keys():
                            prefix = 'predict-tail ' + id2category[i] + ' ' + metric
                            metrics[prefix] = sum([log[metric] for log in logs])/len(logs)
            # pdb.set_trace()
            # save hits dictionary for analysis
            # import pickle
            # f = open("./models/RotatE_wn18rr_0/hits.pkl", "wb")
            # pickle.dump(hits, f)
            # f.close()

        return metrics
