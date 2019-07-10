from config import FLAGS
from utils_siamese import convert_msec_to_sec_str
from time import time
import numpy as np
from collections import OrderedDict, defaultdict


def train_loop(data, model, saver, sess):
    train_costs, train_times, ss, val_results_dict = [], [], [], OrderedDict()
    print('Optimization Started!')
    for iter in range(FLAGS.iters):
        iter += 1
        tvt = get_train_tvt(iter)
        # Train.
        feed_dict = model.get_feed_dict_for_train(data)
        r, train_time = run_tf(
            feed_dict, model, saver, sess, tvt, iter=iter)
        train_s = ''
        train_cost = r
        train_costs.append(train_cost)
        train_times.append(train_time)
        s = 'Iter:{:04n} {} train_loss={:.5f}{} time={}'.format(
            iter, tvt, train_cost, train_s, convert_msec_to_sec_str(train_time))
        print(s)
        ss.append(s)
    print('Optimization Finished!')
    return train_costs, train_times


def test(data, model, saver, sess):
    gs1, gs2 = data.test_gs, data.train_gs + data.val_gs
    node_embs_dict, graph_embs_mat, emb_time = collect_embeddings(
        gs1, gs2, model, saver, sess)
    saver.save_test_info(node_embs_dict, graph_embs_mat, emb_time)


def run_pairs_for_val_test(row_graphs, col_graphs, eval, model, saver,
                           sess, val_or_test, care_about_loss=True):
    m = len(row_graphs)
    n = len(col_graphs)
    sim_dist_mat = np.zeros((m, n))
    time_list = []
    loss_list = []
    print_count = 0
    flush = True
    for i in range(m):
        for j in range(n):
            g1 = row_graphs[i]
            g2 = col_graphs[j]
            if care_about_loss:
                true_sim_dist = eval.get_true_dist_sim(i, j, val_or_test, model)
                if true_sim_dist is None:
                    continue
            else:
                true_sim_dist = 0  # only used for loss
            feed_dict = model.get_feed_dict_for_val_test(g1, g2, true_sim_dist, False)
            (loss_i_j, dist_sim_i_j), test_time = run_tf(
                feed_dict, model, saver, sess, val_or_test)
            if flush:
                (loss_i_j, dist_sim_i_j), test_time = run_tf(
                    feed_dict, model, saver, sess, val_or_test)
                flush = False
            test_time *= 1000
            if val_or_test == 'test' and print_count < 100:
                print('{},{},{:.2f}mec,{:.4f},{:.4f}'.format(
                    i, j, test_time, dist_sim_i_j, true_sim_dist))
                print_count += 1
            sim_dist_mat[i][j] = dist_sim_i_j
            loss_list.append(loss_i_j)
            time_list.append(test_time)
    return sim_dist_mat, loss_list, time_list


def run_tf(feed_dict, model, saver, sess, tvt, iter=None):
    if tvt == 'train':
        objs = [model.opt_op, model.train_loss]
    elif tvt == 'test':
        objs = [model.pred_sim_without_act()]
    elif tvt == 'test_node_emb':
        objs = [model.node_embeddings]
    elif tvt == 'test_graph_emb':
        objs = [model.graph_embeddings_val_test]
    else:
        raise RuntimeError('Unknown train_test {}'.format(tvt))
    objs = saver.proc_objs(objs, tvt, iter)  # may become [loss_related_obj, objs...]
    t = time()
    outs = sess.run(objs, feed_dict=feed_dict)
    time_rtn = time() - t
    saver.proc_outs(outs, tvt, iter)
    if tvt == 'train':
        rtn = outs[-1]
    else:
        rtn = outs[-1]
    return rtn, time_rtn


def collect_embeddings(test_gs, train_gs, model, saver, sess, gemb_only=False):
    assert (hasattr(model, 'node_embeddings'))
    # if not hasattr(model, 'graph_embeddings_val_test'):
    #     return None, None, None
    # [train + val ... test]
    all_gs = train_gs + test_gs
    node_embs_dict = defaultdict(list)  # {0: [], 1: [], ...}
    graph_embs_mat, emb_time = None, None
    if hasattr(model, 'graph_embeddings_val_test'):
        graph_embs_mat = np.zeros((len(all_gs), model.gemb_dim))
    emb_time_list = []
    for i, g in enumerate(all_gs):
        feed_dict = model.get_feed_dict_for_val_test(g, g, 1.0)
        if not gemb_only:
            node_embs, t = run_tf(
                feed_dict, model, saver, sess, 'test_node_emb')
            t *= 1000
            emb_time_list.append(t)
            for gcn_id, node_embs_list_length_two in enumerate(node_embs):
                assert (len(node_embs_list_length_two) == 2)
                node_embs_dict[gcn_id].append(node_embs_list_length_two[0])
        # Only collect graph-level embeddings when the model produces them.
        if hasattr(model, 'graph_embeddings_val_test'):
            graph_embs, _ = run_tf(
                feed_dict, model, saver, sess, 'test_graph_emb')
            assert (len(graph_embs) == 2)
            graph_embs_mat[i] = graph_embs[0]
    if emb_time_list:
        emb_time = np.mean(emb_time_list)
        print('node embedding time {:.5f}msec'.format(emb_time))
    if hasattr(model, 'graph_embeddings_val_test') and not gemb_only:
        print(graph_embs_mat[0:2])
    return node_embs_dict, graph_embs_mat, emb_time


def get_train_tvt(iter):
    tf_opt = 'train'
    return tf_opt


def pretty_print_dict(d, indent=0):
    for key, value in sorted(d.items()):
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty_print_dict(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))
