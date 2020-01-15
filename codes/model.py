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
        
        self.entity_dim = hidden_dim*2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim*2 if double_relation_embedding else hidden_dim

        if model_name.endswith('tRotatE') or model_name.endswith('tRotationH'):
            self.n_tuple = int(model_name[0])
            assert self.n_tuple in [2, 4]
            self.entity_dim *= self.n_tuple

        if model_name == 'LinearTransE':
            self.relation_dim *= 2

        if model_name in ['RotatH', 'RotatTransH']:
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            self.entity_dim += 1  # use the last column as entity bias
            if model_name == 'RotatTransH':
                self.relation_dim += (self.entity_dim-1)

        if model_name == 'RotatTransE':
            self.relation_dim += self.entity_dim


        if model_name.endswith('RotationH') or model_name.endswith('LinearTransH'):
            # deprecated
            self.entity_dim += 1 # use the last column as entity bias
            self.curvature = nn.Parameter(torch.zeros(1, ))
            self.softplus = nn.Softplus()
            if not self.relation_dim % 2 == 0: raise ValueError('dim must be even')
            self.phase_dim = int(self.relation_dim / 2)
            if model_name.endswith('LinearTransH'):
                self.phase_dim *= 4 # 2x2 linear transformation
            self.relation_dim += self.phase_dim #  add phases dim for hyperbolic rotation

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

        if model_name == 'LinearTransE':
            # print('Warning: Using xavier uniform init for relation embedding')
            # nn.init.xavier_uniform_(self.relation_embedding)
            print(colored('!! Warning: initiaze 1st part of relation embedding to be 1 !!', 'red'))
            nn.init.ones_(self.relation_embedding[..., 0:int(self.relation_dim/2.0)])

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', '2tRotatE', '4tRotatE', 'pRotatE', \
                              'RotationH', '2tRotationH', '4tRotationH', 'LinearTransE', 'RotatH', 'RotatTransH', 'RotatTransE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'tRotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('tRotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')
        
    def forward(self, sample, mode='single'):
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
            
        else:
            raise ValueError('mode %s not supported' % mode)
            
        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'RotatTransE': self.RotatE,
            'RotatH': self.RotatH,
            'RotatTransH': self.RotatH,
            'pRotatE': self.pRotatE,
            'RotationH': self.RotationH,
            'LinearTransE': self.LinearTransE,
        }
        
        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        elif self.model_name.endswith('tRotatE'):
            score = self.tRotatE(head, relation, tail, mode)
        elif self.model_name.endswith('tRotationH'):
            score = self.tRotationH(head, relation, tail, mode)
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
        head = hyperbolic.proj(hyperbolic.expmap0(hyperbolic.proj_tan0(head, c), c=c), c=c)
        tail = hyperbolic.proj(hyperbolic.expmap0(hyperbolic.proj_tan0(tail, c), c=c), c=c)

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]
        if self.model_name == 'RotatTransH':
            phase_relation, translation = relation.split([int((self.entity_dim-1)/2.), self.entity_dim-1], dim=2)
            phase_relation = phase_relation / (self.embedding_range.item() / pi)
            translation = hyperbolic.proj(hyperbolic.expmap0(hyperbolic.proj_tan0(translation, c), c=c), c=c)
        else:
            phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation

        res = torch.cat([re_score, im_score], dim=2)

        if self.model_name == 'RotatTransH':
            res = hyperbolic.mobius_add(res, translation, c)

        score = hyperbolic.sqdist(res, tail, c)
        score = bh + bt - score
        return score

    def LinearTransE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # relation = relation.reshape(relation.shape[0], relation.shape[1], re_head.shape[2], 2)

        # scale rotateE
        scale, phase = torch.chunk(relation, 2, dim=2)
        phase_relation = phase / (self.embedding_range.item() / pi)
        a = torch.cos(phase_relation) * scale
        b = - torch.sin(phase_relation) * scale
        d = a
        c = -b

        # normalize matrix (frobenius norm)
        # norm = relation.norm(dim=-1).unsqueeze(3)
        # relation = relation / norm

        # normalize matrix (column norm)
        # norm_1 = relation[..., [0, 2]].norm(dim=-1).unsqueeze(3)
        # norm_2 = relation[..., [1, 3]].norm(dim=-1).unsqueeze(3)
        # relation[..., [0, 2]] = relation[..., [0, 2]] / norm_1
        # relation[..., [1, 3]] = relation[..., [1, 3]] / norm_2

        # a, b, c, d = relation[..., 0], relation[..., 1], relation[..., 2], relation[..., 3]

        # conformal constraint
        # a, b = relation[..., 0], relation[..., 1]
        # d = a
        # c = -b

        if mode == 'head-batch':
            det = 1 / ((a * d - b * c) + 1e-15)
            re_score = det * (d * re_tail - b * im_tail)
            im_score = det * (a * im_tail - c * re_tail)
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * a + im_head * b
            im_score = re_head * c + im_head * d
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    def tRotatE(self, head, relation, tail, mode):

        heads = torch.chunk(head, self.n_tuple, dim=2)
        tails = torch.chunk(tail, self.n_tuple, dim=2)

        # sum the score for each element in the tuple
        score = 0.
        for i in range(self.n_tuple):
            score += self.RotatE(heads[i], relation, tails[i], mode)

        # average the scores
        score /= self.n_tuple

        return score

    def pRotatE(self, head, relation, tail, mode):
        pi = 3.14159262358979323846
        
        #Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head/(self.embedding_range.item()/pi)
        phase_relation = relation/(self.embedding_range.item()/pi)
        phase_tail = tail/(self.embedding_range.item()/pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)            
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim = 2) * self.modulus
        return score

    def RotationH(self, head, relation, tail, mode):
        raise ValueError('This version has to be fixed')
        dim = head.shape[2] - 1

        if mode == 'head_batch':
            head, bh = tail.split([dim, 1], dim=2)
            tail, bt = head.split([dim, 1], dim=2)
        else:
            head, bh = head.split([dim, 1], dim=2)
            tail, bt = tail.split([dim, 1], dim=2)

        trans, phase = relation.split([self.relation_dim-self.phase_dim, self.phase_dim], dim=2)
        c = self.softplus(self.curvature)

        h_head = hyperbolic.expmap0(head, c)
        h_tail = hyperbolic.expmap0(tail, c)
        h_trans = hyperbolic.expmap0(trans, c)

        # rotate
        pi = 3.14159265358979323846
        phase = phase / (self.embedding_range.item() / pi)

        re_head, im_head = torch.chunk(h_head, 2, dim=2)

        re_relation = torch.cos(phase)
        im_relation = torch.sin(phase)

        re_rot = re_relation * re_head + im_relation * im_head
        im_rot = re_relation * im_head - im_relation * re_head

        x = torch.cat([re_rot, im_rot], dim=-1)

        # translate
        x = hyperbolic.mobius_add(x, h_trans, c)

        # distance
        d = hyperbolic.dist(x, h_tail, c)

        # add entity bias
        score = d ** 2 + bh + bt

        return score.squeeze(2)

    def LinearTransH(self, head, relation, tail, mode):

        dim = head.shape[2] - 1

        if mode == 'head_batch':
            head, bh = tail.split([dim, 1], dim=2)
            tail, bt = head.split([dim, 1], dim=2)
        else:
            head, bh = head.split([dim, 1], dim=2)
            tail, bt = tail.split([dim, 1], dim=2)

        translate, transform = relation.split([self.relation_dim - self.phase_dim, self.phase_dim], dim=2)
        c = self.softplus(self.curvature)

        h_head = hyperbolic.expmap0(head, c)
        h_tail = hyperbolic.expmap0(tail, c)
        h_translate = hyperbolic.expmap0(translate, c)

        # linear transformation
        transform = transform.view(transform.shape[0], transform.shape[1], -1, 4)
        re_head, im_head = torch.chunk(h_head, 2, dim=2)

        re_rot = transform[..., 0] * re_head + transform[..., 1] * im_head
        im_rot = transform[..., 2] * im_head + transform[..., 3] * re_head

        x = torch.cat([re_rot, im_rot], dim=-1)

        # translate
        x = hyperbolic.mobius_add(x, h_translate, c)

        # distance
        d = hyperbolic.dist(x, h_tail, c)

        # add entity bias
        score = d ** 2 + bh + bt

        return score.squeeze(2)


    def tRotationH(self, head, relation, tail, mode):
        head, bh = head.split([self.entity_dim - 1, 1], dim=2)
        tail, bt = tail.split([self.entity_dim - 1, 1], dim=2)

        heads = torch.chunk(head, self.n_tuple, dim=2)
        tails = torch.chunk(tail, self.n_tuple, dim=2)

        # sum the score for each element in the tuple
        score = 0.
        for i in range(self.n_tuple):
            heads_i = torch.cat([heads[i], bh], dim=2)
            tails_i = torch.cat([tails[i], bt], dim=2)
            score += self.RotationH(heads_i, relation, tails_i, mode)

        # average the scores
        score /= self.n_tuple

        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        positive_score = model(positive_sample)

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

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
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
                    'head-batch'
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
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), mode)
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
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
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
