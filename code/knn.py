import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos




def construct_graph(dataset, features, topk):
    fname = 'data/' + dataset + '/raw/knn/tmp.txt'
    print(fname)
    f = open(fname, 'w')
    ##### Kernel
    # dist = -0.5 * pair(features) ** 2
    # dist = np.exp(dist)

    #### Cosine
    dist = cos(features)
    inds = []
    for i in range(dist.shape[0]):
        # 取出前topk + 1个最大cos的特征点的序号
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用enumerate
    for i, v in enumerate(inds):
        for vv in v:
            if vv == i:
                pass
            else:
                f.write('{} {}\n'.format(i, vv))
    f.close()


def generate_knn(dataset):
    for topk in range(2, 10):
        data = np.loadtxt('data/' + dataset + '/raw/' + dataset + '.feature', dtype=float)
        print(data)
        construct_graph(dataset, data, topk)
        f1 = open('data/' + dataset + '/raw/knn/tmp.txt','r')
        f2 = open('data/' + dataset + '/raw/knn/c' + str(topk) + '.txt', 'w')
        lines = f1.readlines()
        for line in lines:
            start, end = line.strip('\n').split(' ')
            if int(start) < int(end):
                f2.write('{} {}\n'.format(start, end))
        f2.close()


'''generate KNN graph'''

# generate_knn('RegNetwork')
