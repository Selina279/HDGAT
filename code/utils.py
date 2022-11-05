import numpy as np
import scipy.sparse as sp
from random import shuffle
import torch
from tqdm import tqdm
import os


def load_data(path, dataset):
    print('Loading {} dataset...'.format(dataset))
    type_list = ['tweet', 'emotion', 'topic']
    type_have_label = 'tweet'
    features_list = []
    tidx_map_list = []
    tidx2type = {t: set() for t in type_list}

    for type_name in type_list:
        print('Loading {} content...'.format(type_name))
        features_block = False
        print(path)
        print(dataset)
        print(type_name)
        indexes, features, labels = [], [], []
        with open("{}{}_content_{}.csv".format(path, dataset, type_name), encoding='utf-8') as f:
            if type_name == 'tweet':
                for line in tqdm(f.readlines()[:9291]):
                    cache = line.strip().split(',')
                    indexes.append(np.array(cache[0], dtype=int))
                    labels.append(np.array([cache[3]], dtype=str))
                    features.append(np.array(cache[4:], dtype=np.float32))
                features = np.stack(features)
                features = normalize(features)
            elif type_name == 'topic':
                for line in tqdm(f.readlines()[9291:18205]):
                    cache = line.strip().split(',')
                    indexes.append(np.array(cache[0], dtype=int))
                    labels.append(np.array(cache[3], dtype=str))
                    features.append(np.array(cache[4:], dtype=np.float32))
                features = np.stack(features)
                features = normalize(features)
            else:
                for line in tqdm(f.readlines()[18205:]):
                    cache = line.strip().split(',')
                    indexes.append(np.array(cache[0], dtype=int))
                    labels.append(np.array(cache[3], dtype=str))
                    features.append(np.array(cache[4:], dtype=np.float32))
                features = np.stack(features)
                features = normalize(features)
            if not features_block:
                features = torch.FloatTensor(np.array(features))
                features = dense_tensor_to_sparse(features)

            features_list.append(features)
        print(len(indexes), len(labels), len(features))

        if type_name == type_have_label:
            labels = np.stack(labels)
            print(labels.size)
            labels = encode_onehot(labels)
            print(labels)
        Labels = torch.LongTensor(labels)
        print("label matrix shape: {}".format(Labels.shape))

        tidx = np.stack(indexes)
        for i in tidx:
            tidx2type[type_name].add(i)
        tidx_map = {j: i for i, j in enumerate(tidx)}
        tidx_map_list.append(tidx_map)
        print("done.")
        # feature补零
        len_list = [len(tidx2type[t]) for t in type_list]
        type2len = {t: len(tidx2type[t]) for t in type_list}
        len_all = sum(len_list)
        print(len(len_list))


        print('Building graph...')
        # 生成邻接矩阵
        adj_list = [[None for _ in range(len(type_list))] for __ in range(len(type_list))]
        # 处理边 建图
        edges_unordered = []
        edge_fo = open("{}{}_map.csv".format(path, dataset), 'r', encoding='utf8')
        for line in edge_fo:
            line = line.split()
            edges_unordered.append([w for w in line])
        adj_all = sp.lil_matrix(np.zeros((len_all, len_all)), dtype=np.float32)

        for i1 in range(len(type_list)):
            for i2 in range(len(type_list)):
                t1, t2 = type_list[i1], type_list[i2]
                if i1 == i2:
                    edges = []
                    for edge in edges_unordered:
                        if (edge[0] in tidx2type[t1] and edge[1] in tidx2type[t2]):
                            edges.append([tidx_map_list[i1].get(edge[0]), tidx_map_list[i2].get(edge[1])])
                    edges = np.array(edges)
                    if len(edges) > 0:
                        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                            shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                    else:
                        adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                    print(adj_all)
                    # adj_all[sum(len_list[:i1]): sum(len_list[:i1+1]),
                    #         sum(len_list[:i2]): sum(len_list[:i2+1])] = adj.tolil()

                elif i1 < i2:
                    edges = []
                    for edge in edges_unordered:
                        if (edge[0] in tidx2type[t1] and edge[1] in tidx2type[t2]):
                            edges.append([tidx_map_list[i1].get(edge[0]), tidx_map_list[i2].get(edge[1])])
                        # elif (edge[1] in tidx2type[t1] and edge[0] in tidx2type[t2]):
                        #     edges.append([tidx_map_list[i1].get(edge[1]), tidx_map_list[i2].get(edge[0])])
                    edges = np.array(edges)
                    if len(edges) > 0:
                        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                            shape=(type2len[t1], type2len[t2]), dtype=np.float32)
                    else:
                        adj = sp.coo_matrix((type2len[t1], type2len[t2]), dtype=np.float32)

                    # adj_all[
                    #     sum(len_list[:i1]): sum(len_list[:i1+1]),
                    #     sum(len_list[:i2]): sum(len_list[:i2+1])] = adj.tolil()
                    # adj_all[
                    #     sum(len_list[:i2]): sum(len_list[:i2 + 1]),
                    #     sum(len_list[:i1]): sum(len_list[:i1 + 1])] = adj.T.tolil()

                else:
                    pass
        # 处理邻接矩阵，原始邻接矩阵 - -对成化（加上自连接的邻接矩阵） - -归一化
        adj_all = adj_all + adj_all.T.multiply(adj_all.T > adj_all) - adj_all.multiply(adj_all.T > adj_all)
        adj_all = normalize_adj(adj_all + sp.eye(adj_all.shape[0]))  # 传入normalize中的类型是matrix,输出是matrix

        for i1 in range(len(type_list)):
            for i2 in range(len(type_list)):
                adj_list[i1][i2] = sparse_mx_to_torch_sparse_tensor(
                    adj_all[sum(len_list[:i1]): sum(len_list[:i1 + 1]),
                            sum(len_list[:i2]): sum(len_list[:i2 + 1])]
                )

        print("Num of edges: {}".format(adj_all.getnnz()))  # .nonsero()[0]
        tidx_train, tidx_val, tidx_test = load_divide_tidx(path, tidx_map_list[0])
        return adj_list, features_list, Labels, tidx_train, tidx_val, tidx_test, tidx_map_list[0]


def encode_onehot(labels):
    classes = set(labels.T[0])  # 求出labels的类
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}  # 生成类别字典
    labels_onehot = np.array(list(map(classes_dict.get, labels.T[0])), dtype=np.int32)
    return labels_onehot  # 类型还没加上


def load_divide_tidx(path, tidx_map):
    tidx_train = []
    tidx_val = []
    tidx_test = []
    with open(path + 'train.csv', 'r', encoding='utf-8-sig') as f:
        for line in f:
            tidx_train.append(tidx_map.get(int(line.strip('\n'))))

    print("tidx_train", tidx_train)

    with open(path + 'val.csv', 'r', encoding='utf-8-sig') as f:
        for line in f:
            tidx_val.append(tidx_map.get(int(line.strip('\n'))))
    print("tidx_val", tidx_val)

    with open(path + 'test.csv', 'r', encoding='utf-8-sig') as f:
        for line in f:
            tidx_test.append(tidx_map.get(int(line.strip('\n'))))

    print("tidx_test", tidx_test)
    shuffle(tidx_val)
    # idx_val = idx_val[:80]

    print(len(tidx_train), len(tidx_val), len(tidx_test))
    tidx_train = torch.LongTensor(tidx_train)
    tidx_val = torch.LongTensor(tidx_val)
    tidx_test = torch.LongTensor(tidx_test)
    return tidx_train, tidx_val, tidx_test


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # r_mat_inv类型是dia_matrix
    mx = r_mat_inv.dot(mx)  # 矩阵相乘
    return mx


def normalize_adj(mx):
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor """
    if len(sparse_mx.nonzero()[0]) == 0:
        r, c = sparse_mx.shape
        return torch.sparse.FloatTensor(r, c)
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def dense_tensor_to_sparse(dense_mx):
    return sparse_mx_to_torch_sparse_tensor(sp.coo.coo_matrix(dense_mx))


def accuracy(output, labels):
    preds = output.max(1)[1].types_as(labels)  # 找到每行的最大值（负数）所在列
    correct = preds.eq(labels).double()  # equ()两者相等置1，不等值0
    correct = correct.sum()
    return correct / len(labels)


def resample(train, val, test: torch.LongTensor, path, tidx_map, rewrite = True):
    if os.path.exists(path+'train_inductive.map'):
        rewrite = False
        filenames = ["train", "unlabeled", "vali", "test"]
        ans = []
        for file in filenames:
            with open(path+file+"_inductive.map", "r") as f:
                cache = []
                for line in f:
                    cache.append(tidx_map.get(int(line)))
            ans.append(torch.LongTensor(cache))
        return ans

    tidx_train = train
    tidx_test = val
    cache = list(test.numpy())
    shuffle(cache)
    tidx_val = cache[: tidx_train.shape[0]]
    tidx_unlabeled = cache[tidx_train.shape[0]: ]
    tidx_val = torch.LongTensor(tidx_val)
    tidx_unlabeled = torch.LongTensor(tidx_unlabeled)

    print("\n\ttrain: ", tidx_train.shape[0],
          "\n\tunlabeled: ", tidx_unlabeled.shape[0],
          "\n\tvali: ", tidx_val.shape[0],
          "\n\ttest: ", tidx_test.shape[0])
    if rewrite:
        tidx_map_reverse = dict(map(lambda t: (t[1], t[0]), tidx_map.items()))
        filenames = ["train", "unlabeled", "tidx_val", "tidx_test"]
        ans = [tidx_train, tidx_unlabeled, tidx_val, tidx_test]
        for i in range(4):
            with open(path+filenames[i]+"_inductive.map", "w") as f:
                f.write("\n".join(map(str, map(tidx_map_reverse.get, ans[i].numpy()))))
    return tidx_train, tidx_unlabeled, tidx_val, tidx_test


dataset = 'WU3D'

if __name__ == '__main__':
    # sys.stdout = Logger("{}.log".format(dataset))
    path = '../data/' + dataset + '/'
    adj, features, labels, tidx_train_ori, tidx_val_ori, tidx_test_ori, tidx_map = load_data(path=path, dataset=dataset)
    print(adj)