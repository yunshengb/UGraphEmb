from utils import load, save, create_dir_if_not_exists, get_save_path, load_data
from exp import save_fig, get_color_map, set_save_paths_for_vis
from vis import vis_small
from shapely.geometry import Point, LineString
import numpy as np
from os.path import join
from glob import glob
from sklearn.manifold import TSNE
import matplotlib

# Fix font type for ACM paper submission.
matplotlib.use('Agg')
matplotlib.rc('font', **{'family': 'serif', 'size': 22})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt


class LineInterp(object):
    def __init__(self, x1, y1, x2, y2, num_points):
        assert (num_points >= 2)
        self.x1, self.y1, self.x2, self.y2, self.num_points = \
            x1, y1, x2, y2, num_points

    def name(self):
        return 'line_interp_x1={}_y1={}_x2={}_y2={}_{}ps'.format(
            self.x1, self.y1, self.x2, self.y2, self.num_points)


# DATASET = 'webeasy'
# DIR = '/media/yba/HDD/GraphEmbedding/model/Siamese/exp/stage10:KDD2019_2/siamese_regression_webeasy_2019-01-25T00:00:02.021302(mse; att 2 dense; 0.749; 0.344 -- tsne good)'


# DATASET = 'webeasy'
# COLOR_P, COLOR_M, COLOR_A = 'orangered', 'blue', 'blue'
# DIR = '/media/yba/HDD/GraphEmbedding/model/Siamese/exp/stage9:KDD2019/siamese_regression_dist_webeasy_2019-01-07T02:23:37.496610(mse; att; 0.765; 0.393; emb tsne plot updated)'


# DATASET = 'nci109'
# DIR = '/media/yba/HDD/GraphEmbedding/model/Siamese/exp/stage9:KDD2019/siamese_regression_dist_nci109_2019-01-21T17:12:49.633545(mse; att 2 dense; bs=512; iters=8000; 0.644; 0.705)'

# DATASET = 'ptc'
# COLOR = 'green'
# DIR = '/media/yba/HDD/GraphEmbedding/model/Siamese/exp/stage9:KDD2019/siamese_regression_dist_ptc_2019-01-22T01:52:23.111816(mse; att 2 dense; 0.881; 0.725)'

# DATASET = 'imdbmulti'
# COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
# DIR = '/media/yba/HDD/GraphEmbedding/model/Siamese/exp/stage9:KDD2019/siamese_regression_dist_imdbmulti_2019-01-02T21:53:24.646747(mse; ssrc; 0.932; 0.483)'

# DATASET = 'imdbmulti'
# COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
# DIR = '/home/yba/Documents/GraphEmbedding/result/ijcai_2019_baselines/graphsage_imdb'

# DATASET = 'imdbmulti'
# COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
# DIR = '/home/yba/Documents/GraphEmbedding/result/ijcai_2019_baselines/graph2vec_imdb'

# DATASET = 'imdbmulti'
# COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
# DIR = '/home/yba/Documents/GraphEmbedding/result/ijcai_2019_baselines/netmf_imdb'

DATASET = 'imdbmulti'
COLOR_P, COLOR_M, COLOR_A = 'dodgerblue', 'coral', 'red'
DIR = '/home/yba/Documents/GraphEmbedding/result/ijcai_2019_baselines/graphlet_imdb'

# DATASET = 'nci109'
# COLOR_P, COLOR_M, COLOR_A = 'lightseagreen', 'red', 'red'
# DIR = '/home/yba/Documents/GraphEmbedding/model/Siamese/logs/siamese_regression_nci109_2019-02-16T18:41:14.926233'

# DATASET = 'reddit10k'
# COLOR_P, COLOR_M, COLOR_A = 'blueviolet', 'yellowgreen', 'yellowgreen'
# DIR = '/home/yba/Documents/GraphEmbedding/model/Siamese/logs/siamese_regression_reddit10k(UGEmb_0.400)_2019-02-22T20:51:12.365416'


confs = [LineInterp(x1=12, y1=-18, x2=30, y2=10, num_points=12)]


def main():
    embs = _load_embs()
    points = _to_points(embs)
    for conf in confs:
        _plot_conf(points, conf)


def _load_embs():
    klepto_pickle_dir = join(DIR, 'test_info.klepto')
    if not glob(klepto_pickle_dir):
        pickle_file_list = glob(join(DIR, '*.pickle'))
        print('pickle_file_list', pickle_file_list)
        klepto_pickle_dir = pickle_file_list[0]
        tsned_path = join(DIR, 'tsne_2')
    else:
        tsned_path = join(klepto_pickle_dir, 'tsne_2')
    l = load(tsned_path, use_klepto=False)
    if l is not None:
        print('loaded tsne embs {} from {}'.format(l.shape, tsned_path))
        return l
    l = load(klepto_pickle_dir)
    if not l:
        raise RuntimeError('Not loaded')
    if 'graph_embs_mat' in l:
        orig_embs = l['graph_embs_mat']  # siamese
    elif 'P' in l:
        orig_embs = l['P']  # kernel
    else:
        assert (False)
    print('loaded {}'.format(orig_embs.shape))
    tsne = TSNE(n_components=2)
    embs = tsne.fit_transform(orig_embs)
    print('TSNE embeddings: {} --> {} to plot'.format(
        orig_embs.shape, embs.shape))
    save(tsned_path, embs)
    print('Saved to', tsned_path)
    return embs


def _to_points(embs):
    assert (embs.shape[1] == 2)
    rtn = []
    for emb in embs:
        rtn.append(Point(emb))
    print('Created {} Point objects'.format(len(rtn)))
    return rtn


def _plot_conf(points, conf):
    gs = load_data(DATASET, True).graphs + load_data(DATASET, False).graphs
    line = LineString([(conf.x1, conf.y1), (conf.x2, conf.y2)])
    print(line)
    selected_ps = []
    selected_gs = []
    for perc in _get_percs(conf):
        lp = line.interpolate(perc, normalized=True)
        id, emb_p = _closest_point(lp, points)
        selected_gs.append(gs[id])
        selected_ps.append(emb_p)
    print('plotting...')
    _plot_points(points, [], conf)
    _plot_points(points, selected_ps, conf)
    _plot_gs(selected_gs, conf)


def _plot_gs(selected_gs, conf):
    assert (len(selected_gs) >= 2)
    info_dict = {
        # draw node config
        'draw_node_size': 20,
        'draw_node_label_enable': True,
        'show_labels': False,
        'node_label_type': 'type',
        'node_label_name': 'type',
        'draw_node_label_font_size': 6,
        'draw_node_color_map': get_color_map(selected_gs),
        # draw edge config
        'draw_edge_label_enable': False,
        'draw_edge_label_font_size': 6,
        # graph text info config
        'each_graph_text_list': [],
        'each_graph_text_font_size': 10,
        'each_graph_text_pos': [0.5, 1.05],
        # graph padding: value range: [0, 1]
        'top_space': 0.20,  # out of whole graph
        'bottom_space': 0.05,
        'hbetween_space': 0.6,  # out of the subgraph
        'wbetween_space': 0,
        # plot config
        'plot_dpi': 200,
        'plot_save_path_eps': '',
        'plot_save_path_png': ''
    }
    plt_cnt = 0
    info_dict, plt_cnt = set_save_paths_for_vis(
        info_dict, DIR, None, '{}_{}_gs'.format(DATASET, conf.name()), plt_cnt)
    info_dict['each_graph_text_list'] = [i + 1 for i in range(len(selected_gs))]
    vis_small(selected_gs[0], selected_gs[1:], info_dict)
    print('Saved {} query demo plots'.format(plt_cnt))


def _plot_points(points, selected_ps, conf):
    plt.figure()
    for p in points:
        x = _get_coord(p)[0]
        y = _get_coord(p)[1]
        plt.scatter(x, y, c=COLOR_P, marker='*')
    for i, p in enumerate(selected_ps):
        x = _get_coord(p)[0]
        y = _get_coord(p)[1]
        plt.scatter(x, y, facecolors='none', edgecolors=COLOR_M, s=200)
        # if i == 0 or i == len(selected_ps) - 1:
        plt.annotate(i + 1, (x, y), fontsize=15, color=COLOR_A)
    if not selected_ps:
        plt.grid(linestyle='dashed')
        name = '{}_grid_tsne'.format(DATASET)
    else:
        plt.axis('off')
        cur_axes = plt.gca()
        cur_axes.axes.get_xaxis().set_visible(False)
        cur_axes.axes.get_yaxis().set_visible(False)
        name = '{}_{}_tsne'.format(DATASET, conf.name())
    save_fig(plt, DIR, name, print_path=True)
    plt.close()


def _closest_point(lp, points):
    dists = []
    for point in points:
        dists.append(lp.distance(point))
    id = np.argmin(dists)
    return id, points[id]


def _get_percs(conf):
    return list(np.arange(0, 1, 1 / (conf.num_points - 1))) + [1]


def _get_coord(point):
    return list(point.coords)[0]


if __name__ == '__main__':
    main()
    # print(list(np.arange(0.0, 1.0, 0.1)))
