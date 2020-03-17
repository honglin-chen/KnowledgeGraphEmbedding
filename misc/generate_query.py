#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import sys

sys.path.append('../codes')
from utils import hyperbolic_utils as hyp
from utils import manifolds
from run import *
from adjustText import adjust_text
import networkx as nx
import itertools
from scipy.special import comb
import multiprocessing as mp
import pdb

# # Loading checkpoint

# In[36]:


# data_path = '../data/wn18rr'
# ckpt_path = '../models/RotatCones2_wn18rr_1/checkpoint/ckpt_29999'

# data_path = '../data/synthetic_tree_2'
# ckpt_path = '../models/EmbedCones_synthetic_tree_2_-3/checkpoint/ckpt_9999'
# # filename = 'RotatCones wn18rr_small_2_1 id:4 neg:50 dim: 1'
# filename = 'none'

data_path = '../data/wn18_small_3'
ckpt_path = '../models/EmbedCones_wn18_small_3_-1/checkpoint/ckpt_19999'
# ckpt_path = '../models/PoincareEmbedding_wn18_small_1_-1/checkpoint/ckpt_7999'
# filename = 'RotatCones wn18rr_small_2_1 id:4 neg:50 dim: 1'
filename = 'none'

# data_path = '../data/synthetic_tree_1'
# ckpt_path = '../models/EmbedCones_wn18_small_1_-1/checkpoint/ckpt_9999'
# # filename = 'RotatCones wn18rr_small_2_1 id:4 neg:50 dim: 1'
# filename = 'none'


# data_path = '../data/synthetic_tree_1'
# ckpt_path = '../models/RotatTransH2_synthetic_tree_1_-2/checkpoint/ckpt_999'
# # filename = 'RotatCones wn18rr_small_2_1 id:4 neg:50 dim: 1'
# filename = 'none'

# data_path = '../data/synthetic_tree_1'
# ckpt_path = '../models/PoincareEmbedding_synthetic_tree_1_-2/checkpoint/ckpt_1999'
# # filename = 'RotatCones wn18rr_small_2_1 id:4 neg:50 dim: 1'
# filename = 'none'


# data_path = '../data/wn18rr_small_2_1'
# ckpt_path = '../models/PoincareEmbedding_wn18rr_small_2_1_-2/checkpoint/ckpt_27499'
# filename = 'Poincare Embedding wn18rr_small_2_1 id:-2 neg:50 dim: 1'

# data_path = '../data/wn18rr_small_2'
# ckpt_path = '../models/RotatCones2_wn18rr_small_2_2/checkpoint/ckpt_17499'
# filename = 'current_figure'

# data_path = '../data/wn18rr_small_2'
# ckpt_path = '../models/PoincareEmbedding_wn18rr_small_2_0/checkpoint/ckpt_67499'
# filename = 'Poincare Embedding wn18rr_small_2 id:0 neg:50 dim: 1'

# data_path = '../data/wn18rr'
# ckpt_path = '../models/PoincareEmbedding_wn18rr_0/checkpoint/ckpt_37499'
# filename = 'Poincare Embedding wn18rr id:0 neg:50 dim: 500'


# data_path = '../data/wn18rr'
# ckpt_path = '../models/RotatH_wn18rr_-1/checkpoint/ckpt_79999'


checkpoint = torch.load(ckpt_path, map_location='cpu')
print('Loading checkpoint from {}'.format(ckpt_path))
model_state_dict = checkpoint['model_state_dict']
save_entity_embedding = model_state_dict['entity_embedding']
relation_embedding = model_state_dict['relation_embedding']
c = torch.nn.Softplus()(model_state_dict['curvature'])
embedding_range = model_state_dict['embedding_range']
entity_dim = save_entity_embedding.shape[1]
if entity_dim % 2 == 1: save_entity_embedding = save_entity_embedding[:, 0:entity_dim - 1]


# # Create Graph

G = nx.read_gpickle(os.path.join(data_path, "graph.gpickle"))

with open(os.path.join('../data/wn18_small_3', 'entities.dict')) as fin:
    node2id = dict()
    id2entity = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        node2id[entity] = int(eid)
        id2entity[eid] = entity


# # Generate query

# In[13]:


def lca_helper(pair):
    n1, n2 = pair
    stp_1 = nx.shortest_path_length(G, target=n1)
    stp_2 = nx.shortest_path_length(G, target=n2)
    del stp_1[n1], stp_2[n2]  # delete source node from the dictionary
    intersection = set.intersection(set(stp_1.keys()), stp_2.keys())
    if len(intersection) == 1:
        min_len = [(k, stp_1[k] + stp_2[k]) for k in intersection]
        return (n1, n2, intersection, min_len[0][1])
    elif len(intersection) > 1:
        len_sum = [(k, stp_1[k] + stp_2[k]) for k in intersection]
        min_len = min(len_sum, key=lambda x: x[1])
        lca = [k for k, length in len_sum if length == min_len[1]]
        return (n1, n2, lca, min_len[1])


nodes = G.nodes()
combinations = itertools.combinations(nodes, 2)
print(G.order())
print(comb(G.order(), 2))
print(mp.cpu_count())

pool = mp.Pool(4)

results = []
start = time.time()
results = pool.map_async(lca_helper, combinations)
print('func time: ', time.time() - start)
start = time.time()
results = results.get()
print('result time: ', time.time() - start, 'avg: ', (time.time() - start) / (comb(G.order(), 2)))
start = time.time()
results = list(filter(None, results))
print(len(results))
print('filter time: ', time.time() - start)
pool.close()

# # Evaluate LCA

# In[18]:


lca_query = results



K = 0.1

manifold = manifolds.PoincareManifold(K=K)

entity_embedding = save_entity_embedding[0:-1, :]
print(entity_embedding.shape)

half_aperture = manifold.half_aperture(entity_embedding)

correct_count = 0.
count = 0


def evaluate_single_lca_qeury(lca_query):
    p, q, lca, min_len = lca_query

    p = node2id[str(p)]
    q = node2id[str(q)]
    lca = list(lca)
    lca = [node2id[str(i)] for i in lca]

    # lca = list(lca)
    p_e = entity_embedding[p, :].unsqueeze(0)
    q_e = entity_embedding[q, :].unsqueeze(0)
    lca_e = entity_embedding[lca[0], :]

    score_cones_p = manifold.angle_at_u(entity_embedding, p_e) - half_aperture
    score_cones_q = manifold.angle_at_u(entity_embedding, q_e) - half_aperture
    score_sum = score_cones_p.clamp(min=0.) + score_cones_q.clamp(min=0.)
    # score_sum[mask] = -1.
    if (score_sum == 0).nonzero().shape[0] == 0:
        pred = score_sum.argmin().item()
        # pred = 0
        # if pred not in lca:
        #     print(p, q, lca, pred)
            # print('here')
    else:

        candidate = (score_sum == 0).nonzero()
        candidate_norm = entity_embedding[candidate, :].norm(dim=-1)
        #         avg_norm = 0.5 * (entity_embedding[p, :].norm(dim=-1) + entity_embedding[q, :].norm(dim=-1))
        #         margin = 5e-3
        #         f = (candidate_norm < (avg_norm + margin)) & (candidate_norm > (avg_norm - margin))
        #         candidate_norm[f] = -1

        # for i in range(candidate.shape[0]):
        #     if candidate[i] in leaf_nodes:
        #         candidate_norm[i] = -1.

        pred = candidate[candidate_norm.argmax()].item()

        # if pred not in lca:
        #     # pdb.set_trace()
        #     print(p, q, lca, pred, candidate, candidate_norm, candidate_norm.argmax(), len(candidate))
        #     # print(score_cones_p, score_cones_q)

    # if pred in lca:
    #     correct_count += 1.
    # #     else:
    # #         print(p, q, lca, pred)
    # count += 1
    # if count % 1000 == 0:
    #     print(count, correct_count / count)

    if pred in lca:
        return 1.0
    else:
        return 0.


pool = mp.Pool(4)

results = []
start = time.time()
results = pool.map_async(evaluate_single_lca_qeury, lca_query)
print('func time: ', time.time() - start)
start = time.time()
results = results.get()
print('result time: ', time.time() - start, 'avg: ', (time.time() - start) / (comb(G.order(), 2)))
start = time.time()
# results = list(filter(None, results))
print(len(results))
print(sum(results) / len(lca_query), len(lca_query))
print('filter time: ', time.time() - start)
pool.close()



