from collections import Counter, defaultdict
from copy import deepcopy
from operator import itemgetter

import math
import numpy as np
import sklearn
from sklearn.base import TransformerMixin
from sklearn.neighbors import NearestNeighbors

class GDHS_LC(TransformerMixin):
    def __init__(self,k1, k2, k3, w, **kwargs) -> None:
        """
        Parameters
        ----------
        k1,k2 synthetic parameter
        w weight coefficient between safe factor and generalization factor
        θ parameter for cleaning overlapping between majority class and other majority classes
        θ=0.8,w=0.8,k1=5,k2=5,k3=5
        """
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.w = w
        self.quantities, self.goal_quantity = [None] * 2

    def fit_transform(self, _X, _y, shuffle: bool = False):

        X = deepcopy(_X)
        y = deepcopy(_y)
        ori_X = X.tolist()   #preserve original data

        assert len(X.shape) == 2, 'X should have 2 dimension'
        assert X.shape[0] == y.shape[0], 'Number of labels must be equal to number of samples'

        self.quantities = Counter(y)
        self.goal_quantity = round(len(y)/len(self.quantities))

        dsc_maj_cls = sorted(((v, i) for v, i in self.quantities.items() if i >= self.goal_quantity), key=itemgetter(1),
                             reverse=True)
        asc_min_cls = sorted(((v, i) for v, i in self.quantities.items() if i < self.goal_quantity), key=itemgetter(1),
                             reverse=False)

        record_weight = defaultdict(float)  #record selection weight for undersampling process
        for i,sample_id in enumerate(ori_X):
            record_weight[i] = 0.0

        for class_name, class_quantity in asc_min_cls:
            X, y, record_weight = self._oversample(X, y, ori_X, class_name, record_weight)

        for class_name, class_quantity in dsc_maj_cls:
            X, y = self._undersample_maj_min(X, y, ori_X, class_name, record_weight)

        if len(dsc_maj_cls) > 1:
            for class_name_maj, class_quantity_maj in dsc_maj_cls:
                X, y = self._undersample_maj_maj(X, y, class_name_maj, dsc_maj_cls)

        if shuffle:
            X, y = sklearn.utils.shuffle(X, y)

        return np.array(X), np.array(y)

    def _undersample_maj_maj(self, X, y, this_class, dsc_maj_cls):
        indices_in_class = [i for i, value in enumerate(y) if value == this_class]

        maj_name = []
        for class_name, class_quantity in dsc_maj_cls:
            if class_name != this_class:
                maj_name.append(class_name)

        indices_all_maj_class = []
        for class_name, class_quantity in dsc_maj_cls:
            for i, value in enumerate(y):
                if value == class_name:
                    indices_all_maj_class.append(i)

        n_neigh = min(len(indices_all_maj_class),self.k1)
        neigh_clf = NearestNeighbors(n_neighbors=n_neigh).fit(X)
        nei_dis, neighbor_indices = neigh_clf.kneighbors(X[indices_in_class])

        count_diff = dict.fromkeys(indices_in_class, 0)
        for i, sample_id in enumerate(indices_in_class):
            for num in range(n_neigh):
                if y[neighbor_indices[i][num]] != this_class and y[neighbor_indices[i][num]] in maj_name:
                    count_diff[sample_id] = count_diff[sample_id] + 1

        remove_indices = []
        for key, value in count_diff.items():
            if value >= 4: #θ=0.8
                remove_indices.append(key)
            else:
                continue

        if len(remove_indices) > 0:
            X = np.delete(X, remove_indices, axis=0)
            y = np.delete(y, remove_indices, axis=0)

        return X, y

    def _undersample_maj_min(self, X, y, ori_X, this_class, record_weight):
        indices_in_class = [i for i, value in enumerate(y) if value == this_class]

        record_maj_weight = defaultdict(float)
        for i, sample_index in enumerate(indices_in_class):
            record_maj_weight[sample_index] = 0.0
        for i, sample_index in enumerate(indices_in_class):
            for ori_i, sample_id in enumerate(ori_X):
                if all(ori_X[ori_i] == X[sample_index]):
                    record_maj_weight[sample_index] = record_weight[ori_i]
                    break

        weight_mean = np.mean(list(record_maj_weight.values()))
        remove_indices = []
        for key, value in record_maj_weight.items():
            if value > weight_mean:
                remove_indices.append(key)
            else:
                continue

        if len(remove_indices) > 0:
            X = np.delete(X, remove_indices, axis=0)
            y = np.delete(y, remove_indices, axis=0)

        return X, y


    def _oversample(self, X, y, ori_X, class_name, record_weight):
        this_class = class_name

        indices_in_class = [i for i, value in enumerate(y) if value == class_name]

        indices_all_else_class = []
        for i, value in enumerate(y):
            if value != this_class:
                indices_all_else_class.append(i)

        pos = X[indices_in_class]
        neg = X[indices_all_else_class]
        label_neg = y[indices_all_else_class]
        num_need_create = max(0, self.goal_quantity - (len(indices_in_class)))

        y_train_pos = y[indices_in_class]
        k_weight = self.k3
        attr = pos.shape[1]

        factor,record_maj_weight = self.oversample_select_weight(X, y, ori_X, this_class, record_weight)

        new_factor = list()
        for i in range(len(factor)):
            factor_ = factor[i] / np.sum(factor[i])
            new_factor.append(factor_)
        new_factor = np.array(new_factor)

        # fitting the model
        n_neigh = min(len(pos), k_weight)
        if n_neigh == 1:
            n_neigh = n_neigh + 1
        else:
            n_neigh = n_neigh
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(pos)
        dist, ind = nn.kneighbors(pos)

        # generating samples
        new = list()
        base_indices = np.random.choice(list(range(len(pos))), num_need_create)  # num_create_final

        for j in range(len(base_indices)):
            final_factor = new_factor[base_indices[j]]

            neighbor_indices = np.random.choice(list(range(1, n_neigh)), p=final_factor)
            x_base = pos[base_indices[j]]
            x_neighbor = pos[ind[base_indices[j], neighbor_indices]]

            giff = np.random.rand(attr)

            samples = x_base + np.multiply((x_neighbor - x_base), giff)

            new.append(samples)

        x_final = np.vstack((pos, np.array(new)))

        if num_need_create == 0:
            x_label_final = y_train_pos
        else:
            x_label_final = np.hstack((y_train_pos, np.hstack([this_class] * num_need_create)))

        x_train_final = np.vstack((x_final, neg))
        x_train_label_final = np.hstack((x_label_final, label_neg))

        return x_train_final, x_train_label_final,record_maj_weight

    def oversample_select_weight(self, X, y, ori_X, class_name, record_weight):
        safe_score = self.oversample_safe_factor(X, y, class_name)
        safe_score = np.array(safe_score)

        k_weight = self.k3
        w1 = self.w

        indices_in_class = [i for i, value in enumerate(y) if value == class_name]
        pos = X[indices_in_class]

        indices_all_else_class = []

        for i, value in enumerate(y):
            if value != class_name:
                indices_all_else_class.append(i)
        neg = X[indices_all_else_class]

        # fitting the model
        n_neigh = min([len(pos), k_weight])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(pos)
        dist, ind = nn.kneighbors(pos)

        safe_factor = [[] for _ in range(len(pos))]
        generalization_factor = [[] for _ in range(len(pos))]

        for element_3 in range(len(ind)):
            distance__ = dist[element_3]
            distance_ = distance__[1:n_neigh]
            max_dist = distance_[-1]

            n_neigh_neg = len(neg)
            nn_neg = NearestNeighbors(n_neighbors=n_neigh_neg, n_jobs=1)
            nn_neg.fit(neg)
            dist_neg, ind_neg = nn_neg.kneighbors([pos[element_3]])

            dist_i = [i for item in dist_neg for i in item]
            ind_i = [i for item in ind_neg for i in item]

            list_neg = list()
            for element_4 in range(len(dist_i)):
                if dist_i[element_4] <= max_dist:
                    sam = neg[ind_i[element_4]]
                    list_neg.append(sam)
            array_neg = np.array(list_neg)

            neighbor_ind = ind[element_3]
            neighbor_ind = neighbor_ind[1:n_neigh]

            if len(array_neg) == 0:
                for i in range(n_neigh - 1):
                    generalization_factor[element_3].append(math.exp(0))

                    relative_score_i = safe_score[neighbor_ind[i]]
                    safe_factor[element_3].append(np.exp(relative_score_i))
            else:
                nbr_samples = pos[neighbor_ind]

                area_ = distance_
                area = area_ / np.sum(area_)
                for element_5 in range(len(nbr_samples)):
                    seed_samples = pos[element_3]
                    assist_samples = nbr_samples[element_5]

                    attr_min = [[] for _ in range(len(seed_samples))]
                    attr_max = [[] for _ in range(len(seed_samples))]
                    for attr in range(len(seed_samples)):
                        min_attr = min(seed_samples[attr], assist_samples[attr])
                        max_attr = max(seed_samples[attr], assist_samples[attr])
                        attr_min[attr].append(min_attr)
                        attr_max[attr].append(max_attr)

                    select_neg = list()
                    for sam in array_neg:
                        evalu = list()
                        for att in range(len(sam)):
                            if attr_min[att] <= sam[att] <= attr_max[att]:
                                evalu.append(1)
                        if len(evalu) == len(sam):
                            select_neg.append(sam)

                    if len(select_neg) == 0:
                        density_ratio = 0
                        v1 = 1
                        generalization_ = 1 / (np.exp(density_ratio + v1))
                        generalization_factor[element_3].append(generalization_)

                        relative_score_i = safe_score[neighbor_ind[element_5]]
                        safe_factor[element_3].append(np.exp(relative_score_i))

                    elif len(select_neg) == 1:
                        num_factor = len(select_neg) / len(array_neg)
                        distance_factor = area[element_5]
                        density_ratio = num_factor / (distance_factor + 1)
                        v1 = 0
                        generalization_ = 1 / (np.exp(density_ratio + v1))
                        generalization_factor[element_3].append(generalization_)

                        relative_score_i = safe_score[neighbor_ind[element_5]]
                        safe_factor[element_3].append(np.exp(relative_score_i))

                        for sam in select_neg:
                            for i_1, sample_id in enumerate(ori_X):
                                if all(sample_id == sam):
                                    break
                            record_weight[i_1] = record_weight[i_1] + ((1 - w1) * generalization_) + (w1 * (np.exp(relative_score_i)))

                    else:

                        num_factor = len(select_neg) / len(array_neg)
                        distance_factor = area[element_5]
                        density_ratio = num_factor / (distance_factor + 1)

                        attr1_min = [[] for _ in range(len(seed_samples))]
                        attr1_max = [[] for _ in range(len(seed_samples))]
                        for attr1 in range(len(seed_samples)):
                            attr_list = list()
                            for i in select_neg:
                                attr_list.append(i[attr1])

                            max_attr1 = max(attr_list)
                            min_attr1 = min(attr_list)
                            attr1_min[attr1].append(min_attr1)
                            attr1_max[attr1].append(max_attr1)
                        attr_min = np.array(attr_min).flatten()
                        attr_max = np.array(attr_max).flatten()
                        attr1_min = np.array(attr1_min).flatten()
                        attr1_max = np.array(attr1_max).flatten()

                        v1 = 1
                        for i in range(len(attr_max)):
                            range_s = abs(attr_max[i] - attr_min[i])
                            v_neg = abs(attr1_max[i] - attr1_min[i])
                            if range_s == 0 or v_neg == 0:
                                v1 = v1 * 1
                            else:
                                v1 = v1 * (v_neg / range_s)
                        generalization_ = 1 / (1 + density_ratio + v1)
                        generalization_factor[element_3].append(generalization_)

                        relative_score_i = safe_score[neighbor_ind[element_5]]
                        safe_factor[element_3].append(np.exp(relative_score_i))

                        for sam in select_neg:
                            for i_2, sample_id in enumerate(ori_X):
                                if all(sample_id == sam):
                                    break
                            record_weight[i_2] = record_weight[i_2] + ((1 - w1) * generalization_) + (w1 * (np.exp(relative_score_i)))

        factor = ((1 - w1) * np.array(generalization_factor)) + (w1 * np.array(safe_factor))
        return factor,record_weight

    def oversample_safe_factor(self, X, y, class_name):
        this_class = class_name
        x_train = X
        y_train = y
        k_count = self.k2

        indices_in_class = [i for i, value in enumerate(y) if value == class_name]
        pos = X[indices_in_class]

        # fitting the model
        n_neigh = min([len(pos), k_count])
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=1)
        nn.fit(x_train)
        dist, ind = nn.kneighbors(pos)
        sf_lis = []

        for element_1 in range(len(ind)):
            ind_1 = ind[element_1]
            ind_new = ind_1[1:]
            num_same = 0
            for i in ind_new:
                if y_train[i] == this_class:
                    num_same = num_same + 1
            sf_lis.append(num_same / n_neigh)
        return sf_lis
