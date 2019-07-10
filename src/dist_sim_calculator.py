from utils import get_save_path, save, load, load_data, get_norm_str, create_dir_if_not_exists
from dist_sim import normalized_dist_sim
from collections import OrderedDict
import numpy as np


class DistSimCalculator(object):
    def __init__(self, dataset, ds_metric, algo):
        if ds_metric == 'ged':
            self.dist_sim_func = None
            ds = 'dist'
        else:
            raise RuntimeError('Unknwon distance/similarity metric {}'.format(ds_metric))
        self.sfn = '{}/{}_{}_{}{}_gidpair_{}_map'.format(
            get_save_path(), dataset, ds_metric, algo,
            '' if algo == 'astar' or algo == 'graphlet' or algo == 'mccreesh2017' else '_revtakemin', ds)
        self.algo = algo
        self.gidpair_ds_map = load(self.sfn)
        if not self.gidpair_ds_map:
            self.gidpair_ds_map = OrderedDict()
            save(self.sfn, self.gidpair_ds_map)
            print('Saved dist/sim map to {} with {} entries'.format(
                self.sfn, len(self.gidpair_ds_map)))
        else:
            print('Loaded dist/sim map from {} with {} entries'.format(
                self.sfn, len(self.gidpair_ds_map)))

    def calculate_dist_sim(self, g1, g2, dec_gsize=False, return_neg1=False):
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        d = self.gidpair_ds_map.get(pair)
        if d is None:
            rev_pair = (gid2, gid1)
            rev_d = self.gidpair_ds_map.get(rev_pair)
            if rev_d:
                d = rev_d
            else:
                if return_neg1:
                    return -1, -1
                print('calc...', gid1, gid2)
                raise NotImplementedError()
                d = self.dist_sim_func(g1, g2, self.algo)
                if self.algo != 'astar':
                    d = min(d, self.dist_sim_func(g2, g1, self.algo))
            self.gidpair_ds_map[pair] = d
            print('{}Adding entry ({}, {}) to dist map'.format(
                ' ' * 80, pair, d))
            # save(self.sfn, self.gidpair_ds_map)
        return d, normalized_dist_sim(d, g1, g2, dec_gsize=dec_gsize)

    def load(self, row_gs, col_gs, mats=None, csv_filenames=None, ds_metric=None,
             check_symmetry=False):
        """
        Load the internal distance map from external distance matrices,
            each of which is assumed to be m by n, or external csv files.
        Use this function if the pairwise distances have been calculate
            elsewhere, e.g. by the multiprocessing version of
            running the baselines as in one of the functions in exp.py.
        The distance map stored in this distance calculator will be
            enriched/expanded by the results.
        Be careful of the inputs! Check row_gs, col_gs, and the actual matrices
            or csv files match!
        :param row_gs: the corresponding row graphs
        :param col_gs: the corresponding column graphs
        :param mats:
        :param csv_filename:
        :param ds_metric: currently only support ged
        :param check_symmetry: whether to check if mat if symmetric or not
        :return:
        """
        exsiting_entries_list = None
        if mats:
            m, n = mats[0].shape
            assert (m == len(row_gs) and n == len(col_gs))
        else:
            assert (csv_filenames)
            print('Loading', csv_filenames)
            exsiting_entries_list = [load_from_exsiting_csv(csv_filename, ds_metric)
                                     for csv_filename in csv_filenames]
            print('Done loading with {} extries'.format([len(x) for x in exsiting_entries_list]))
        m, n = len(row_gs), len(col_gs)
        valid_count = 0
        for i in range(m):
            # print(i, i / m)
            for j in range(n):
                i_gid = row_gs[i].graph['gid']
                j_gid = col_gs[j].graph['gid']
                d = self._min_vote_from(i, j, i_gid, j_gid, mats, exsiting_entries_list, ds_metric)
                print('d', d)
                if check_symmetry:
                    d_t = self._min_vote_from(j, i, j_gid, i_gid, mats, exsiting_entries_list, ds_metric)
                    if d != d_t:
                        raise RuntimeError(
                            'Asymmetric distance {} {}: {} and {}'.format(
                                i, j, d, d_t))
                gid1 = row_gs[i].graph['gid']
                gid2 = col_gs[j].graph['gid']
                pair = (gid1, gid2)
                d_m = self.gidpair_ds_map.get(pair)
                if d_m is not None and d != -1:
                    # print('@')
                    if d != d_m:
                        # print('#')
                        print(
                            'Inconsistent distance {} {}: {} and {}'.format(
                                i, j, d, d_m))
                else:
                    print('*')
                    if d != -1:
                        print('))')
                        valid_count += 1
                        self.gidpair_ds_map[pair] = d
                # if d == -1:
                #     exit(-1)
        save(self.sfn, self.gidpair_ds_map)
        print('{} valid entries loaded; {} entries in map'.format(
            valid_count, len(self.gidpair_ds_map)))

    def _min_vote_from(self, i, j, i_gid, j_gid, mats=None, exsiting_entries_list=None, ds_metric=None):
        """
        Assume the min needs to be taken. For MCS, may need the max (to be implemented).
        :param i:
        :param j:
        :param mats:
        :param exsiting_entries_list:
        :param ds_metric:
        :return:
        """
        if ds_metric == 'ged':
            func = np.min
        elif ds_metric == 'mcs' or ds_metric == 'glet':
            func = np.max
        else:
            raise NotImplementedError()
        if mats:
            return func([mat[i][j] for mat in mats])
        else:
            assert (exsiting_entries_list)
            cands = []
            for exsiting_entries in exsiting_entries_list:
                tmp = exsiting_entries.get((i_gid, j_gid))
                if tmp is not None:
                    if ds_metric == 'ged':
                        print('@@@', tmp)
                        _, _, _, _, ds, _, _ = tmp
                    else:
                        print('###', tmp)
                        _, _, _, _, ds, _, _, _ = tmp
                    cands.append(ds)
                else:
                    print('!!!', tmp)
            if len(cands) != len(exsiting_entries_list):  # not all finish computing for this pair
                print(i, j, i_gid, j_gid, len(cands), len(exsiting_entries_list), len(exsiting_entries_list[0]))
                # raise RuntimeError()
                return -1  # -1 indicates invalid ds
            else:
                print(i, j, i_gid, j_gid, len(cands), len(exsiting_entries_list), len(exsiting_entries_list[0]))
                return func(cands)

    def add_one_entry(self, g1, g2, ds_true, save):
        """
        :param g1:
        :param g2:
        :param ds_true: Assume to be the true dist/sim.
        Dangerous to use if this is not the case. Would pollute the map.
        :return:
        """
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        ds = self.gidpair_ds_map.get(pair)
        if ds is None:
            if ds_true < 0:
                raise ValueError('Cannot add {} < 0'.format(ds_true))
            self.gidpair_ds_map[pair] = ds_true
        else:
            if ds != ds_true:
                raise ValueError(
                    '{} aleady in the map {} and {} != {}'.format(
                        pair, ds, ds, ds_true))
            print('{} aleady in the map {}; skip'.format(pair, ds))
        if save:
            save(self.sfn, self.gidpair_ds_map)


def get_train_train_dist_mat(dataset, dist_metric, dist_algo, norm):
    train_data = load_data(dataset, train=True)
    gs = train_data.graphs
    dist_sim_calculator = DistSimCalculator(dataset, dist_metric, dist_algo)
    return get_gs_ds_mat(gs, gs, dist_sim_calculator, 'train', 'train', dataset,
                         dist_metric, dist_algo, norm)


def get_gs_ds_mat(gs1, gs2, dist_sim_calculator, tvt1, tvt2,
                  dataset, dist_metric, dist_algo, norm, dec_gsize, return_neg1=False):
    mat_str = '{}({})_{}({})'.format(tvt1, len(gs1), tvt2, len(gs2))
    dir = '{}/ds_mat'.format(get_save_path())
    create_dir_if_not_exists(dir)
    sfn = '{}/{}_{}_ds_mat_{}{}_{}'.format(
        dir, dataset, mat_str, dist_metric,
        get_norm_str(norm), dist_algo)
    l = load(sfn)
    if l is not None:
        print('Loaded from {}'.format(sfn))
        return l
    m = len(gs1)
    n = len(gs2)
    dist_mat = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            g1 = gs1[i]
            g2 = gs2[j]
            d, normed_d = dist_sim_calculator.calculate_dist_sim(
                g1, g2, dec_gsize=dec_gsize, return_neg1=return_neg1)
            if norm:
                dist_mat[i][j] = normed_d
            else:
                dist_mat[i][j] = d
    save(sfn, dist_mat)
    print('Saved to {}'.format(sfn))
    return dist_mat

