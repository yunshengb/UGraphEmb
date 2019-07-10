from config import FLAGS
from model import Model
from samplers import SelfShuffleList
from utils_siamese import get_coarsen_level, get_flags, is_transductive
from inits import glorot
from dist_sim_kernel import create_ds_kernel
from dist_sim_calculator import get_gs_ds_mat
import numpy as np
import tensorflow as tf


class SiameseRegressionModel(Model):
    def __init__(self, input_dim, data, dist_sim_calculator):
        self.input_dim = input_dim
        print('original_input_dim', self.input_dim)
        if is_transductive():
            self._create_transductive_gembs_placeholders(data,
                                                         FLAGS.batch_size, FLAGS.batch_size)
        else:
            self._create_basic_placeholders(FLAGS.batch_size, FLAGS.batch_size,
                                            level=get_coarsen_level())
        self.train_y_true = tf.placeholder(
            tf.float32, shape=(FLAGS.batch_size, 1))
        self.val_test_y_true = tf.placeholder(
            tf.float32, shape=(1, 1))
        # Build the model.
        super(SiameseRegressionModel, self).__init__()
        self.ds_kernel = create_ds_kernel(
            FLAGS.ds_kernel, get_flags('yeta'), get_flags('scale'))
        self.train_triples = self._load_train_triples(data, dist_sim_calculator)

    def pred_sim_without_act(self):
        return self.val_test_pred_score

    def apply_final_act_np(self, score):
        return score

    def get_feed_dict_for_train(self, data):
        rtn = {}
        pairs = []
        y_true = np.zeros((FLAGS.batch_size, 1))
        glabels = None
        for i in range(FLAGS.batch_size):
            g1, g2, true_sim_dist = self._sample_train_pair(data)
            pairs.append((g1, g2))
            y_true[i] = true_sim_dist
        rtn[self.train_y_true] = y_true
        rtn[self.dropout] = FLAGS.dropout
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'train')

    def get_feed_dict_for_val_test(self, g1, g2, true_sim_dist):
        rtn = {}
        pairs = [(g1, g2)]
        y_true = np.zeros((1, 1))
        y_true[0] = true_sim_dist
        rtn[self.val_test_y_true] = y_true
        return self._supply_laplacians_etc_to_feed_dict(rtn, pairs, 'val_test')

    def get_true_dist_sim(self, i, j, true_result):
        assert (true_result.dist_or_sim() in ['sim', 'dist'])  # A* GED or MCS
        _, ds = true_result.dist_sim(i, j, FLAGS.ds_norm)
        if FLAGS.supply_sim_dist == 'sim' and true_result.dist_or_sim() == 'dist':  # e.g. A*
            assert (FLAGS.ds_metric == 'ged')
            rtn = self.ds_kernel.dist_to_sim_np(ds)
        elif FLAGS.supply_sim_dist == 'dist' and true_result.dist_or_sim() == 'sim':  # e.g. MCS
            assert (FLAGS.ds_metric == 'mcs')
            raise NotImplementedError()  # TODO
        else:
            rtn = ds
        return rtn

    def _get_ins(self, layer, tvt):
        if is_transductive():
            return self._get_ins_for_transductive_model(layer, tvt)
        else:
            ins = []
            assert (layer.__class__.__name__ == 'GraphConvolution' or
                    layer.__class__.__name__ == 'GraphConvolutionAttention')
            for features in (self._get_plhdr('features_1', tvt) +
                             self._get_plhdr('features_2', tvt)):
                ins.append(features)
        return ins

    def _get_ins_for_transductive_model(self, layer, tvt):
        assert (layer.__class__.__name__ == 'Dist' or
                layer.__class__.__name__ == 'Dot')
        ids_1 = self._get_plhdr('gemb_lookup_ids_1', tvt)
        ids_2 = self._get_plhdr('gemb_lookup_ids_2', tvt)
        gembs_1 = tf.nn.embedding_lookup(self.all_gembs, ids_1)
        gembs_2 = tf.nn.embedding_lookup(self.all_gembs, ids_2)
        if tvt == 'train':
            self.graph_embeddings_train = self._stack_concat([gembs_1, gembs_2])
        elif tvt == 'val_test':
            self.graph_embeddings_val_test = self._stack_concat(
                [gembs_1, gembs_2])  # for train.py to collect
        self.gemb_dim = FLAGS.gemb_dim
        return [gembs_1, gembs_2]

    def _proc_ins_for_merging_layer(self, ins, _):
        assert (len(ins) % 2 == 0)
        proc_ins = []
        i = 0
        j = len(ins) // 2
        for _ in range(len(ins) // 2):
            proc_ins.append([ins[i], ins[j]])
            i += 1
            j += 1
        return proc_ins

    def _val_test_pred_score(self):
        self.val_test_output = self._stack_concat(self.val_test_output)
        assert self.val_test_output.get_shape().as_list() == [1, 1], \
            self.val_test_output.get_shape()
        return tf.squeeze(self.val_test_output)

    def _task_loss(self, tvt):
        rtn = {}
        for loss_lambda in self._get_loss_lambdas_flags():
            if loss_lambda == 'lambda_mse_loss':
                # If the model predicts sim, ground-truth should be sim.
                # If the model predicts dist, ground-truth should be dist.
                assert (FLAGS.supply_sim_dist == FLAGS.pred_sim_dist)
                y_pred, y_true = self._get_y_pred_y_true(tvt)
                loss = tf.nn.l2_loss(y_true - y_pred)
                rtn['mse_loss'] = FLAGS.lambda_mse_loss * loss
            elif loss_lambda == 'lambda_weighted_dist_loss':
                # The model must predict dist and ground-truth must be sim.
                assert (FLAGS.pred_sim_dist == 'dist')
                assert (FLAGS.supply_sim_dist == 'sim')
                assert (FLAGS.ds_metric == 'ged')
                y_pred, y_true_sim = self._get_y_pred_y_true(tvt)
                assert (y_true_sim.get_shape() == y_pred.get_shape())
                loss = tf.reduce_mean(y_true_sim * y_pred)
                rtn['weighted_dist_loss'] = FLAGS.lambda_weighted_dist_loss * loss
            elif loss_lambda == 'lambda_triv_avoid_loss':
                phi, bs_times_2, D = self._get_graph_embs_as_one_mat(tvt)
                loss = tf.reduce_mean(phi ** 2)  # tf.nn.l2_loss does not do sqrt
                rtn['triv_avoid_loss'] = FLAGS.lambda_triv_avoid_loss * loss
            elif loss_lambda == 'lambda_diversity_loss':
                phi, bs_times_2, D = self._get_graph_embs_as_one_mat(tvt)
                # phi /= bs_times_2
                phi_T_phi = tf.matmul(tf.transpose(phi), phi)
                should = tf.eye(D)
                assert (phi_T_phi.get_shape() == should.get_shape())
                loss = tf.reduce_mean((phi_T_phi - should) ** 2)  # tf.nn.l2_loss does not do sqrt
                rtn['diversity_loss'] = FLAGS.lambda_diversity_loss * loss
            elif loss_lambda == 'lambda_gc_loss':
                pass  # handled by model
            else:
                raise RuntimeError('Unknown loss lambda flag {}'.format(loss_lambda))
        return rtn

    def _get_y_pred_y_true(self, tvt):
        if tvt == 'train':
            y_pred = self._stack_concat(self.train_outputs)
            y_true = self.train_y_true
        else:
            y_pred = self._stack_concat(self.val_test_output)
            y_true = self.val_test_y_true
        assert (y_true.get_shape() == y_pred.get_shape())
        return y_pred, y_true

    def _get_graph_embs_as_one_mat(self, tvt):
        phi = self._get_graph_embds(tvt, True)
        s = phi.get_shape().as_list()
        bs_times_2, D = s[0], s[1]
        bs = FLAGS.batch_size if tvt == 'train' else 1
        assert (bs_times_2 == bs * 2)
        return phi, bs_times_2, D

    def _get_graph_embds(self, tvt, need_concat):
        if tvt == 'train':
            rtn = self.graph_embeddings_train
        else:
            assert (tvt == 'val')
            rtn = self.graph_embeddings_val_test
        if need_concat:
            return self._stack_concat(rtn)
        else:
            return rtn

    def _create_basic_placeholders(self, num1, num2, level):
        self.laplacians_1 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num1)]
        self.laplacians_2 = \
            [[[tf.sparse_placeholder(tf.float32)]
              for _ in range(level)] for _ in range(num2)]
        self.features_1 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num1)]
        self.features_2 = \
            [tf.sparse_placeholder(tf.float32) for _ in range(num2)]
        self.num_nonzero_1 = \
            [tf.placeholder(tf.int32) for _ in range(num1)]
        self.num_nonzero_2 = \
            [tf.placeholder(tf.int32) for _ in range(num2)]
        self.edge_index_1 = [tf.sparse_placeholder(tf.int32, shape=(None, 2)) for _ in range(num1)]
        self.edge_index_2 = [tf.sparse_placeholder(tf.int32, shape=(None, 2)) for _ in range(num2)]
        self.incidence_mat_1 = [tf.sparse_placeholder(tf.int32) for _ in range(num1)]
        self.incidence_mat_2 = [tf.sparse_placeholder(tf.int32) for _ in range(num2)]
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.val_test_laplacians_1 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        self.val_test_laplacians_2 = [[[tf.sparse_placeholder(tf.float32)] for _ in range(level)]]
        self.val_test_features_1 = [tf.sparse_placeholder(tf.float32)]
        self.val_test_features_2 = [tf.sparse_placeholder(tf.float32)]
        self.val_test_num_nonzero_1 = [tf.placeholder(tf.int32)]
        self.val_test_num_nonzero_2 = [tf.placeholder(tf.int32)]
        self.val_test_edge_index_1 = [tf.sparse_placeholder(tf.int32, shape=(None, 2))]
        self.val_test_edge_index_2 = [tf.sparse_placeholder(tf.int32, shape=(None, 2))]
        self.val_test_incidence_mat_1 = [tf.sparse_placeholder(tf.int32)]
        self.val_test_incidence_mat_2 = [tf.sparse_placeholder(tf.int32)]

    def _create_transductive_gembs_placeholders(self, data, num1, num2):
        # Create the dataset-level graph-level embeddings table for later look up.
        self.all_gembs = glorot([data.num_graphs(), FLAGS.gemb_dim],
                                name='all_graph_embeddings')
        self.gemb_lookup_ids_1 = tf.placeholder(tf.int32, shape=(num1))
        self.gemb_lookup_ids_2 = tf.placeholder(tf.int32, shape=(num2))
        self.val_test_gemb_lookup_ids_1 = tf.placeholder(tf.int32,
                                                         shape=(1))  # 1 graph pair per val/test
        self.val_test_gemb_lookup_ids_2 = tf.placeholder(tf.int32, shape=(1))
        self.dropout = tf.placeholder_with_default(0., shape=())

    def _load_train_triples(self, data, ds_calc):
        triples = []
        triples = self._load_real_pairs(data.train_gs, data.train_gs,
                                        'train', 'train', triples, ds_calc)
        if is_transductive():
            # Load more pairs to better train the model,
            # since it directly optimizes over the embeddings.
            triples = self._load_real_pairs(data.val_gs, data.train_gs,
                                            'val', 'train', triples, ds_calc)
            triples = self._load_real_pairs(data.val_gs, data.val_gs,
                                            'val', 'val', triples, ds_calc)
            triples = self._load_real_pairs(data.test_gs, data.train_gs,
                                            'test', 'train', triples, ds_calc)
            triples = self._load_real_pairs(data.test_gs, data.val_gs,
                                            'test', 'val', triples, ds_calc)
        return SelfShuffleList(triples)

    def _load_real_pairs(self, gs1, gs2, tvt1, tvt2, triples, ds_calc):
        rtn = []
        nx_gs1 = self._get_nxgraph_list(gs1)
        nx_gs2 = self._get_nxgraph_list(gs2)
        ds_mat = get_gs_ds_mat(
            nx_gs1, nx_gs2, ds_calc, tvt1, tvt2,
            FLAGS.dataset_train, FLAGS.ds_metric, FLAGS.ds_algo, FLAGS.ds_norm,
            dec_gsize=FLAGS.supersource, return_neg1=True)
        m, n = ds_mat.shape
        # assert (m == n)
        # ds_mat_idx = np.argsort(ds_mat, axis=1)
        valid_count, skip_count = 0, 0
        for i in range(m):
            g1 = gs1[i]
            for j in range(n):
                col = j
                g2, ds = gs2[col], ds_mat[i][col]
                if ds < 0:
                    skip_count += 1
                    if tvt1 == 'test':
                        print('@@@', i, j, g1.nxgraph.graph['gid'], g2.nxgraph.graph['gid'],
                              len(gs1), len(gs2))
                        exit()
                    continue
                valid_count += 1
                if FLAGS.ds_metric == 'mcs' or FLAGS.ds_metric == 'glet':
                    assert (FLAGS.supply_sim_dist == 'sim')  # TODO: transform mcs to dist if needed
                    need = ds
                else:
                    assert (FLAGS.ds_metric == 'ged')
                    if FLAGS.supply_sim_dist == 'sim':  # supply for train --> consistent with supply_sim_dist
                        need = self.ds_kernel.dist_to_sim_np(ds)
                    else:
                        need = ds
                rtn.append((g1, g2, need))
        if FLAGS.train_real_percent < 1:
            sp = int(len(rtn) * FLAGS.train_real_percent)
            rtn_new = rtn[0:sp]
            print('Only take {} from {} due to {} percent'.format(
                len(rtn_new), len(rtn), FLAGS.train_real_percent))
            rtn = rtn_new
        triples += rtn
        print('{} valid pairs; {} pairs with dist or sim < 0; {} total triples'.format(
            valid_count, skip_count, len(triples)))
        return triples

    def _sample_train_pair(self, data):
        x, y, true_sim_dist = self.train_triples.get_next()
        return x, y, true_sim_dist

    def _supply_laplacians_etc_to_feed_dict(self, feed_dict, pairs, tvt):
        if is_transductive():
            gemb_lookup_ids_1 = []
            gemb_lookup_ids_2 = []
            for (g1, g2) in pairs:
                gemb_lookup_ids_1.append(g1.global_id)
                gemb_lookup_ids_2.append(g2.global_id)
            feed_dict[self._get_plhdr('gemb_lookup_ids_1', tvt)] = \
                gemb_lookup_ids_1
            feed_dict[self._get_plhdr('gemb_lookup_ids_2', tvt)] = \
                gemb_lookup_ids_2
        else:
            for i, (g1, g2) in enumerate(pairs):
                feed_dict[self._get_plhdr('features_1', tvt)[i]] = \
                    g1.get_node_inputs()
                feed_dict[self._get_plhdr('features_2', tvt)[i]] = \
                    g2.get_node_inputs()
                feed_dict[self._get_plhdr('num_nonzero_1', tvt)[i]] = \
                    g1.get_node_inputs_num_nonzero()
                feed_dict[self._get_plhdr('num_nonzero_2', tvt)[i]] = \
                    g2.get_node_inputs_num_nonzero()
                feed_dict[self._get_plhdr('edge_index_1', tvt)[i]] = \
                    g1.edge_index
                feed_dict[self._get_plhdr('edge_index_2', tvt)[i]] = \
                    g2.edge_index
                feed_dict[self._get_plhdr('incidence_mat_1', tvt)[i]] = \
                    g1.incidence_mat
                feed_dict[self._get_plhdr('incidence_mat_2', tvt)[i]] = \
                    g2.incidence_mat
                assert g1.incidence_mat is not None
                num_laplacians = 1
                for j in range(get_coarsen_level()):
                    for k in range(num_laplacians):
                        feed_dict[
                            self._get_plhdr('laplacians_1', tvt)[i][j][k]] = \
                            g1.get_laplacians(j)[k]
                        feed_dict[
                            self._get_plhdr('laplacians_2', tvt)[i][j][k]] = \
                            g2.get_laplacians(j)[k]
        return feed_dict

    def _get_loss_lambdas_flags(self):
        rtn = []
        d = FLAGS.flag_values_dict()
        for k in d.keys():
            if 'lambda_' in k:
                flag_split = k.split('_')
                assert (flag_split[0] == 'lambda')
                assert (flag_split[-1] == 'loss')
                rtn.append(k)
        return rtn

    def _get_nxgraph_list(self, model_graphs):
        return [g.nxgraph for g in model_graphs]
