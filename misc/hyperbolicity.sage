import networkx as nx
import numpy as np
import os
import pickle as pkl
import time
import pdb
# import utils.data_utils as data_utils
import data_utils


def hyperbolicity_sample(G):
    curr_time = time.time()
    #sp = dict(nx.shortest_path_length(G, weight=None))

    num_samples = 5000
    hyps = []
    for i in range(num_samples):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        #d01 = sp[node_tuple[0]][node_tuple[1]] if node_tuple[0] in sp and node_tuple[1] in sp[node_tuple[0]] else 0
        #d23 = sp[node_tuple[2]][node_tuple[3]] if node_tuple[2] in sp and node_tuple[3] in sp[node_tuple[2]] else 0
        #d02 = sp[node_tuple[0]][node_tuple[2]] if node_tuple[0] in sp and node_tuple[2] in sp[node_tuple[0]] else 0
        #d13 = sp[node_tuple[1]][node_tuple[3]] if node_tuple[1] in sp and node_tuple[3] in sp[node_tuple[1]] else 0
        #d03 = sp[node_tuple[0]][node_tuple[3]] if node_tuple[0] in sp and node_tuple[3] in sp[node_tuple[0]] else 0
        #d12 = sp[node_tuple[1]][node_tuple[2]] if node_tuple[1] in sp and node_tuple[2] in sp[node_tuple[1]] else 0
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2.)
            # print((s[-1] - s[-2]) / 2.)
        except Exception as e:
            continue
    #print('Time for hyp: ', time.time() - curr_time)
    try:
        return max(hyps)

    except:
        return -1


def hyp_ppi():
    data_path = 'data/human_ppi'
    dataset_str = 'human_ppi'
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

def hyp_airport():
    data_path = 'data/airport'
    dataset_str = 'airport'
    graph = pkl.load(open(os.path.join(data_path, dataset_str + '.p'), 'rb'))
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

def hyp_syn():
    data_path = 'data/synthetic_tree_disease_propagation'
    dataset_str = 'synthetic_tree_disease_propagation'
    adj, feat, label = data_utils.load_synthetic_data(dataset_str, False, data_path)
    graph = nx.from_scipy_sparse_matrix(adj)
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    hyp = hyperbolicity_sample(graph)
    print('Hyp: ', hyp)

def hyp_cora():
    data_path = './data/cora'
    dataset_str = 'cora'
    output = data_utils.load_citation_data(dataset_str, False, data_path)
    adj = output[0]
    graph = nx.from_scipy_sparse_matrix(adj)
    print('Computing hyperbolicity', graph.number_of_nodes(), graph.number_of_edges())
    # hyp = hyperbolicity_sample(graph)

    pdb.set_trace()





    print('Hyp: ', hyp)


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def hyp_general(dataset='wn18'):
    print('dataset: %s' % dataset)
    data_dir = './data/%s' % dataset
    with open(os.path.join(data_dir, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(data_dir, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)
    train_triples = read_triple(os.path.join(data_dir, 'train.txt'), entity2id, relation2id)
    valid_triples = read_triple(os.path.join(data_dir, 'valid.txt'), entity2id, relation2id)
    test_triples = read_triple(os.path.join(data_dir, 'test.txt'), entity2id, relation2id)
    all_true_triples = train_triples + valid_triples + test_triples
    print(len(all_true_triples))

    from sage.graphs.hyperbolicity import hyperbolicity
    G = Graph(loops=True)
    pdb.set_trace()
    for i in all_true_triples: G.add_edge(i)
    hyperbolicity(G)

    # # entire graph
    # G = nx.Graph()
    # node_set = []
    # for triple in all_true_triples:
    #     node_set.append(triple[0])
    #     node_set.append(triple[2])
    #     G.add_edge(triple[0], triple[2])
    # node_set = set(node_set)
    # for i in set(node_set):
    #     G.add_node(i)
    #
    # hyp = hyperbolicity_sample(G)
    # print('| entire G |  %5d  |  %5d  | %d  |' % (G.number_of_nodes(), G.number_of_edges(), hyp))
    #
    # for relation in range(0, 18):
    #
    #     # print('------- %d --------' % relation)
    #     G = nx.Graph()
    #     node_set = []
    #     for triple in all_true_triples:
    #         if triple[1] == relation:
    #             node_set.append(triple[0])
    #             node_set.append(triple[2])
    #
    #             G.add_edge(triple[0], triple[2])
    #     for i in set(node_set):
    #         G.add_node(i)
    #     hyp = hyperbolicity_sample(G)
    #     print('|  %d  |  %5d  |  %5d  | %d  |' %(relation, G.number_of_nodes(), G.number_of_edges(), hyp))



if __name__ == '__main__':
    #hyp_airport()
    # hyp_syn()
    #hyp_ppi()
    hyp_general('wn18')
    # hyp_cora()
