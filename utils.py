import torch
import logging
import os
import dgl
import numpy as np
import scipy.sparse as sp

from collections import defaultdict


def set_logger(args):
    if not os.path.exists(os.path.join(args['save_dir'], args['dataset'], args['save_name'])):
        os.makedirs(os.path.join(os.getcwd(), args['save_dir'], args['dataset'], args['save_name']))

    log_file = os.path.join(args['save_dir'], args['dataset'], args['save_name'], 'log.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def read_id(path):
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            tmp[e] = int(t)
    return tmp


def load_triple(path, e2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            h, r, t = e2id[h], r2id[r], e2id[t]
            head.append(h)
            e_type.append(r)
            tail.append(t)
    return head, e_type, tail


def load_ET(path, e2id, t2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t] + len(e2id)
            head.append(e)
            tail.append(t)
            e_type.append(r2id['type'])
    return head, e_type, tail


def load_TYPE_CLU(path, e2id, t2id, r2id, c2id):
    head = []
    t_cluster = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            t, c = line.strip().split('\t')
            t, c = t2id[t] + len(e2id), c2id[c] + len(e2id) + len(t2id)
            head.append(t)
            tail.append(c)
            t_cluster.append(r2id['type_cluster'])
    return head, t_cluster, tail


def load_ENTITY_CLU(path, e2id, t2id, r2id, c2id):
    head = []
    e_cluster = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, c = line.strip().split('\t')
            e, c = e2id[e], c2id[c] + len(e2id) + len(t2id)
            head.append(e)
            tail.append(c)
            e_cluster.append(r2id['entity_cluster'])

    return head, e_cluster, tail


def load_labels(paths, e2id, t2id):
    labels = torch.zeros(len(e2id), len(t2id))
    for path in paths:
        with open(path, encoding='utf-8') as r:
            for line in r:
                e, t = line.strip().split('\t')
                e_id, t_id = e2id[e], t2id[t]
                labels[e_id, t_id] = 1
    return labels


def load_id(path, e2id):
    ret = set()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            ret.add(e2id[e])
    return list(ret)


def load_graph(data_dir, e2id, r2id, t2id, c2id, loadET=True, loadKG=True, loadTC=True, loadEC=True):
    train_label = load_labels([os.path.join(data_dir, 'ET_train.txt')], e2id, t2id)
    train_id = train_label.sum(1).nonzero().squeeze()
    valid_id = load_id(os.path.join(data_dir, 'ET_valid.txt'), e2id)
    test_id = load_id(os.path.join(data_dir, 'ET_test.txt'), e2id)
    all_true = load_labels([
        os.path.join(data_dir, 'ET_train.txt'),
        os.path.join(data_dir, 'ET_valid.txt'),
        os.path.join(data_dir, 'ET_test.txt'),
    ], e2id, t2id).half()
    if loadKG:
        head1, e_type1, tail1 = load_triple(os.path.join(data_dir, 'KG_train.txt'), e2id, r2id)
    else:
        head1, e_type1, tail1 = [], [], []

    if loadET:
        head2, e_type2, tail2 = load_ET(os.path.join(data_dir, 'ET_train.txt'), e2id, t2id, r2id)
    else:
        head2, e_type2, tail2 = [], [], []
    if loadTC:
        head3, e_type3, tail3 = load_TYPE_CLU(os.path.join(data_dir, 'type2cluster.txt'), e2id, t2id, r2id, c2id)
    else:
        head3, e_type3, tail3 = [], [], []
    if loadEC:
        head4, e_type4, tail4 = load_ENTITY_CLU(os.path.join(data_dir, 'entity2cluster.txt'), e2id, t2id, r2id, c2id)
    else:
        head4, e_type4, tail4 = [], [], []
    head = torch.LongTensor(head1 + head2 + head3 + head4 + tail1 + tail2 + tail3 + tail4)
    tail = torch.LongTensor(tail1 + tail2 + tail3 + tail4 + head1 + head2 + head3 + head4)
    g = dgl.graph((head, tail))
    e_type1 = torch.LongTensor(e_type1)
    e_type2 = torch.LongTensor(e_type2)
    e_type3 = torch.LongTensor(e_type3)
    e_type4 = torch.LongTensor(e_type4)
    e_type = torch.cat([e_type1, e_type2, e_type3, e_type4, e_type1 + len(r2id), e_type2 + len(r2id), e_type3 + len(r2id), e_type4 + len(r2id)], dim=0)
    g.edata['etype'] = e_type
    g.ndata['id'] = torch.arange(len(e2id) + len(t2id) + len(c2id))

    return g, train_label, all_true, train_id, valid_id, test_id


def build_sparse_relational_graph(relation_dict, n_entities, n_nodes):
    def _si_norm_lap(adj):
        rowsum = np.array(adj.sum(1))
        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    adj_mat_list = []
    for r_id in relation_dict.keys():
        np_mat = np.array(relation_dict[r_id])
        if r_id == 0:
            cf = np_mat.copy()
            cf[:, 1] = cf[:, 1] + n_entities  # [0, n_types) -> [n_users, n_entities+n_types)
            vals = [1.] * len(cf)
            adj = sp.coo_matrix((vals, (cf[:, 0], cf[:, 1])), shape=(n_nodes, n_nodes))
        else:
            vals = [1.] * len(np_mat)
            adj = sp.coo_matrix((vals, (np_mat[:, 0], np_mat[:, 1])), shape=(n_nodes, n_nodes))
        adj_mat_list.append(adj)

    mean_mat_list = [_si_norm_lap(mat) for mat in adj_mat_list]
    mean_mat_list[0] = mean_mat_list[0].tocsr()[:n_entities, n_entities:].tocoo()

    return mean_mat_list[0]


def build_e2t_graph(args, e2id, t2id):
    relation_dict = defaultdict(list)
    data_path = args["data_dir"] + '/' + args["dataset"] + '/ET_train.txt'
    with open(data_path, "r") as f_reader:
        for line in f_reader.readlines():
            split_line = line.strip().split("\t")
            relation_dict[0].append([int(e2id[split_line[0]]), int(t2id[split_line[1]])])

    t2t_graph = build_sparse_relational_graph(relation_dict, len(e2id), len(e2id) + len(t2id))
    coo = t2t_graph.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    if args['cuda']:
        t2t_graph = torch.sparse.FloatTensor(i, v, coo.shape).to('cuda')
    else:
        t2t_graph = torch.sparse.FloatTensor(i, v, coo.shape)

    t2t_graph_new = t2t_graph
    indice_old = t2t_graph_new._indices()
    value_old = t2t_graph_new._values()
    x = indice_old[0, :]
    y = indice_old[1, :]
    x_A = x
    y_A = y + len(e2id)
    x_A_T = y + len(e2id)
    y_A_T = x
    x_new = torch.cat((x_A, x_A_T), dim=-1)
    y_new = torch.cat((y_A, y_A_T), dim=-1)
    indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
    value_new = torch.cat((value_old, value_old), dim=-1)
    interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
        [len(e2id) + len(t2id), len(e2id) + len(t2id)]))

    return interact_graph


def build_t2c_graph(args, t2id, c2id):
    relation_dict = defaultdict(list)
    data_path = args["data_dir"] + '/' + args["dataset"] + '/type2cluster.txt'
    with open(data_path, "r") as f_reader:
        for line in f_reader.readlines():
            split_line = line.strip().split("\t")
            relation_dict[0].append([int(t2id[split_line[0]]), int(c2id[split_line[1]])])

    t2c_graph = build_sparse_relational_graph(relation_dict, len(t2id), len(t2id) + len(c2id))
    coo = t2c_graph.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    if args['cuda']:
        t2c_graph = torch.sparse.FloatTensor(i, v, coo.shape).to('cuda')
    else:
        t2c_graph = torch.sparse.FloatTensor(i, v, coo.shape)

    t2c_graph_new = t2c_graph
    indice_old = t2c_graph_new._indices()
    value_old = t2c_graph_new._values()
    x = indice_old[0, :]
    y = indice_old[1, :]
    x_A = x
    y_A = y + len(t2id)
    x_A_T = y + len(t2id)
    y_A_T = x
    x_new = torch.cat((x_A, x_A_T), dim=-1)
    y_new = torch.cat((y_A, y_A_T), dim=-1)
    indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
    value_new = torch.cat((value_old, value_old), dim=-1)
    interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
        [len(t2id) + len(c2id), len(t2id) + len(c2id)]))

    return interact_graph


def build_e2c_graph(args, e2id, c2id):
    relation_dict = defaultdict(list)
    data_path = args["data_dir"] + '/' + args["dataset"] + '/entity2cluster.txt'
    with open(data_path, "r") as f_reader:
        for line in f_reader.readlines():
            split_line = line.strip().split("\t")
            relation_dict[0].append([int(e2id[split_line[0]]), int(c2id[split_line[1]])])

    t2c_graph = build_sparse_relational_graph(relation_dict, len(e2id), len(e2id) + len(c2id))
    coo = t2c_graph.tocoo()
    i = torch.LongTensor([coo.row, coo.col])
    v = torch.from_numpy(coo.data).float()
    if args['cuda']:
        t2c_graph = torch.sparse.FloatTensor(i, v, coo.shape).to('cuda')
    else:
        t2c_graph = torch.sparse.FloatTensor(i, v, coo.shape)

    t2c_graph_new = t2c_graph
    indice_old = t2c_graph_new._indices()
    value_old = t2c_graph_new._values()
    x = indice_old[0, :]
    y = indice_old[1, :]
    x_A = x
    y_A = y + len(e2id)
    x_A_T = y + len(e2id)
    y_A_T = x
    x_new = torch.cat((x_A, x_A_T), dim=-1)
    y_new = torch.cat((y_A, y_A_T), dim=-1)
    indice_new = torch.cat((x_new.unsqueeze(dim=0), y_new.unsqueeze(dim=0)), dim=0)
    value_new = torch.cat((value_old, value_old), dim=-1)
    interact_graph = torch.sparse.FloatTensor(indice_new, value_new, torch.Size(
        [len(e2id) + len(c2id), len(e2id) + len(c2id)]))

    return interact_graph


def evaluate(path, predict, all_true, e2id, t2id):
    logs = []
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t]
            tmp = predict[e] - all_true[e]
            tmp[t] = predict[e, t]
            argsort = torch.argsort(tmp, descending=True)
            ranking = (argsort == t).nonzero()
            assert ranking.size(0) == 1
            ranking = ranking.item() + 1
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HIT@1': 1.0 if ranking <= 1 else 0.0,
                'HIT@3': 1.0 if ranking <= 3 else 0.0,
                'HIT@10': 1.0 if ranking <= 10 else 0.0
            })
    MRR = 0
    for metric in logs[0]:
        tmp = sum([_[metric] for _ in logs]) / len(logs)
        if metric == 'MRR':
            MRR = tmp
        logging.debug('%s: %f' % (metric, tmp))
    return MRR


def cal_loss(predict, label, beta):
    loss = torch.nn.BCELoss(reduction='none')
    output = loss(predict, label)
    positive_loss = output * label
    negative_weight = predict.detach()
    negative_weight = beta * (negative_weight - negative_weight.pow(2)) * (1 - label)
    negative_loss = negative_weight * output
    return positive_loss.mean(), negative_loss.mean()
