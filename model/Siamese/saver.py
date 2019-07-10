from config import FLAGS
from utils import get_ts, create_dir_if_not_exists, save_as_dict
from utils_siamese import get_siamese_dir, get_model_info_as_str
import tensorflow as tf
from glob import glob
from os import system
from collections import OrderedDict
from pprint import pprint
from os.path import join


class Saver(object):
    def __init__(self, sess=None):
        model_str = self._get_model_str()
        self.logdir = '{}/logs/{}_{}'.format(
            get_siamese_dir(), model_str, get_ts())
        create_dir_if_not_exists(self.logdir)
        if sess is not None:
            self.tw = tf.summary.FileWriter(self.logdir + '/train', sess.graph)
            self.all_merged = tf.summary.merge_all()
            self.loss_merged = tf.summary.merge(
                self._extract_loss_related_summaries_as_list())
        self._log_model_info(self.logdir, sess)
        self.f = open('{}/results_{}.txt'.format(self.logdir, get_ts()), 'w')
        print('Logging to {}'.format(self.logdir))

    def get_log_dir(self):
        return self.logdir

    def proc_objs(self, objs, tvt, iter):
        if 'train' in tvt:
            objs.insert(0, self.loss_merged)
        return objs

    def proc_outs(self, outs, tvt, iter):
        if 'train' in tvt:
            self.tw.add_summary(outs[0], iter)

    def save_test_info(self, node_embs_dict, graph_embs_mat, emb_time):
        sfn = '{}/test_info'.format(self.logdir)
        # The following function call must be made in one line!
        save_as_dict(sfn, node_embs_dict, graph_embs_mat, emb_time)

    def save_conf_code(self, conf_code):
        with open(join(self.logdir, 'config.py'), 'w') as f:
            f.write(conf_code)

    def save_overall_time(self, overall_time):
        self._save_to_result_file(overall_time, 'overall time')

    def clean_up_saved_models(self, best_iter):
        for file in glob('{}/models/*'.format(self.get_log_dir())):
            if str(best_iter) not in file:
                system('rm -rf {}'.format(file))

    def _get_model_str(self):
        li = []
        key_flags = [FLAGS.model, FLAGS.dataset_train]
        if FLAGS.dataset_val_test != FLAGS.dataset_train:
            key_flags.append(FLAGS.dataset_val_test)
        for f in key_flags:
            li.append(str(f))
        return '_'.join(li)

    def _log_model_info(self, logdir, sess):
        model_info_table = [["**key**", "**value**"]]
        with open(logdir + '/model_info.txt', 'w') as f:
            s = get_model_info_as_str(model_info_table)
            f.write(s)
        model_info_op = \
            tf.summary.text(
                'model_info', tf.convert_to_tensor(model_info_table))
        if sess is not None:
            self.tw.add_summary(sess.run(model_info_op))

    def _save_to_result_file(self, obj, name):
        if type(obj) is dict or type(obj) is OrderedDict:
            # self.f.write('{}:\n'.format(name))
            # for key, value in obj.items():
            #     self.f.write('\t{}: {}\n'.format(key, value))
            pprint(obj, stream=self.f)
        else:
            self.f.write('{}: {}\n'.format(name, obj))

    def _extract_loss_related_summaries_as_list(self):
        rtn = []
        for tensor in tf.get_collection(tf.GraphKeys.SUMMARIES):
            # Assume "loss" is in the loss-related summary tensors.
            if 'loss' in tensor.name:
                rtn.append([tensor])
        return rtn

    def _bool_to_str(self, b, s):
        assert (type(b) is bool)
        if b:
            return s
        else:
            return 'NO{}'.format(s)
