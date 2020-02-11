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
            self.relation_dim += 1

        if model_name in ['RotatCones', 'RotatCones2', 'PoincareEmbedding']:
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.relation_dim += self.entity_dim

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

        # Initialize outside of the K ball, but inside the unit ball.

        if model_name in ['RotatCones', 'RotatCones2']:

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

            self.embedding_range = nn.Parameter(torch.Tensor([0.2]), requires_grad=False)
            nn.init.uniform_(tensor=self.entity_embedding,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())
            nn.init.uniform_(tensor=self.relation_embedding,
                             a=-self.embedding_range.item(),
                             b=self.embedding_range.item())


            with torch.no_grad():
                embedding = self.entity_embedding.data
                embedding = torch.stack(torch.chunk(embedding, 2, dim=1), dim=2)
                norms = embedding.norm(dim=-1, keepdim=True)
                embedding = embedding / (norms + 1e-15) # unit norm
                rand_norm = torch.empty(norms.shape).uniform_(self.inner_radius + self.EPS, self.inner_radius + self.EPS * 2)
                embedding *= rand_norm
                embedding = embedding.view(nentity, self.entity_dim)

                self.entity_embedding.data = embedding

            print('K: %.5f, embedding_range: %.5f, embedding_min: %.5f, embedding_max: %.5f' % \
                  (self.K, self.embedding_range, self.inner_radius + self.EPS, self.inner_radius + self.EPS * 2))



            # poincare_init_path = './models/PoincareEmbedding_wn18rr_0/checkpoint/ckpt_39999'
            # ckpt = torch.load(poincare_init_path)
            # self.load_state_dict(ckpt['model_state_dict'])
            # print('Init checkpoint from %s' % poincare_init_path)
            #
            # resc_vecs = 0.7
            #
            # with torch.no_grad():
            #     entity_embedding_data = self.entity_embedding.data.clone()
            #     translation_embedding_data = self.relation_embedding.data.clone()[:, -self.entity_dim:]
            #     c = self.softplus(self.curvature)
            #
            #     entity_embedding_data = hyperbolic.proj(hyperbolic.expmap0(entity_embedding_data, c), c)
            #     translation_embedding_data = hyperbolic.proj(hyperbolic.expmap0(translation_embedding_data, c), c)
            #
            #     entity_embedding_data *= resc_vecs
            #     translation_embedding_data *= resc_vecs
            #
            #     entity_embedding_data = hyperbolic.logmap0(entity_embedding_data, c)
            #     translation_embedding_data = hyperbolic.logmap0(translation_embedding_data, c)
            #     self.entity_embedding.data = entity_embedding_data
            #     self.relation_embedding.data[:, -self.entity_dim:] = translation_embedding_data
            #
            #
            # # clip vectors
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
                              'expRotatTransH', 'RotatCones', 'RotatCones2', 'PoincareEmbedding', 'RotatHdegree', 'RotatHR']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'tRotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('tRotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')

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
            'RotatCones2': self.RotatCones2,
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

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        # pdb.set_trace()
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation = relation.split([int(self.entity_dim/2.), self.entity_dim], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(translation, c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)

        # print(relation[0,:])
        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        # print(relation[0, :].data.tolist(), phase_relation[0, :].item(), re_relation[0, :].item(), im_relation[0, :].item())

        if mode == 'head-batch':
            raise ValueError

        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

            res = torch.cat([re_score, im_score], dim=2)

            if translation_flag:
                res = hyperbolic.proj(res, c=c)
                res = hyperbolic.mobius_add(res, translation, c)
            res = hyperbolic.proj(res, c=c)
            score = hyperbolic.sqdist(res, tail, c)

        score = (score.squeeze(2) + 1e-4) ** 0.5

        if torch.max(score) == float('inf'):
            pdb.set_trace()

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
        bh = bh.squeeze(2)
        bt = bt.squeeze(2)

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
            res = hyperbolic.mobius_add(res, translation, c)

            energy_radius = bh + bt - hyperbolic.sqdist(hyperbolic.proj(res,c), tail, c)

            res = torch.stack(torch.chunk(res, 2, dim=2), dim=3)
            tail = torch.stack([re_tail, im_tail], dim=3)

            tail = self.clip_vector(tail)
            res = self.clip_vector(res)

            energy_1 = self.score_cones(res, tail)
            energy_2 = self.score_cones(tail, res)


        one_one_mask = (relation_category == 0).unsqueeze(1).unsqueeze(2)
        one_many_mask = (relation_category == 1).unsqueeze(1).unsqueeze(2)
        many_one_mask = (relation_category == 2).unsqueeze(1).unsqueeze(2)

        score_radius = one_one_mask * energy_radius
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

        head = hyperbolic.proj(hyperbolic.expmap0(head, c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(tail, c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if translation_flag:
            phase_relation, translation = relation.split([int(self.entity_dim/2.), self.entity_dim], dim=2)
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
            res = hyperbolic.mobius_add(res, translation, c)

            energy_0 = 6.0 - hyperbolic.sqdist(hyperbolic.proj(res,c), tail, c)

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
    def train_step(model, optimizer, train_iterator, args, step):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()

        if args.train_with_degree:
            positive_sample, negative_sample, subsampling_weight, mode, degree = next(train_iterator)
        else:
            degree = None
            positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()
            degree = None if degree is None else degree.cuda()

        if args.gamma > 1.0:
            print('warning: setting the right gamma')
            pdb.set_trace()

        if args.model in ['RotatCones', 'RotatCones2']:

            negative_energy = model((positive_sample, negative_sample), mode=mode)

            if isinstance(negative_energy, tuple):
                negative_energy_1, negative_energy_2 = negative_energy

                negative_score_1 = - F.logsigmoid(-negative_energy_1.squeeze(dim=2)).mean(dim=1)
                negative_score_2 = (args.gamma - negative_energy_2.clamp(min=0)).clamp(min=0).mean(dim=2)
                negative_score_2 = negative_score_2.mean(dim=1)

                one_one_mask = (positive_sample[:, 3] == 0)
                if args.model == 'RotatCones': # consider radius for all relations
                    negative_score = negative_score_1 + ~one_one_mask * negative_score_2
                else:
                    negative_score = one_one_mask * negative_score_1 + ~one_one_mask * negative_score_2
            else:
                negative_score = (args.gamma - negative_energy).clamp(min=0).mean(dim=2)
                negative_score = negative_score.sum(dim=1)

            positive_energy = model(positive_sample)

            if isinstance(positive_energy, tuple):
                positive_energy_1, positive_energy_2 = positive_energy
                positive_score_1 = - F.logsigmoid(positive_energy_1.sum(dim=2)).squeeze(dim = 1)
                positive_score_2 = positive_energy_2.clamp(min=0).mean(dim=2).squeeze(dim=1)
                # positive_score_2 = - F.logsigmoid(positive_energy_2).mean(dim=2).squeeze(dim=1)

                if args.model == 'RotatCones':
                    positive_score = positive_score_1 + ~one_one_mask * positive_score_2
                else:
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

        elif args.model == 'PoincareEmbedding':
            negative_score = model((positive_sample, negative_sample), mode=mode)
            positive_score = model(positive_sample)
            score = torch.cat([positive_score, negative_score], dim=1)
            # if step % 1 == 0:
            #     print(torch.min(positive_score).item(), torch.min(negative_score).item())
            #     print(torch.max(positive_score).item(), torch.max(negative_score).item())
            loss = F.cross_entropy(score.neg(), torch.zeros(score.shape[0],).long().cuda())
            # pdb.set_trace()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isnan(loss):
                pdb.set_trace()
            loss.backward()
            positive_sample_loss = torch.zeros_like(loss)
            negative_sample_loss = torch.zeros_like(loss)
            regularization_log = {}

        else:
            negative_score = model((positive_sample, negative_sample), mode=mode, degree=degree)

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
        # burnin_step = 2000
        # lr_multilier = 0.1
        # if args.model == 'PoincareEmbedding' and step < burnin_step:
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
                            score_0 = score[0].squeeze(2)
                            score_1 = score[1].mean(dim=2)
                            score_1 = - score_1

                            score_0 += filter_bias
                            # score_1 += (-100 * filter_bias)
                            score_1 += (100 * filter_bias)

                            argsort_0 = torch.argsort(score_0, dim = 1, descending=True)
                            argsort_1 = torch.argsort(score_1, dim = 1, descending=True)

                            one_one_mask = (category == 0).unsqueeze(1)
                            argsort = one_one_mask * argsort_0 + ~one_one_mask * argsort_1


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
