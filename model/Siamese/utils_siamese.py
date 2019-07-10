from config import FLAGS
import sys
from os.path import dirname, abspath, join
import tensorflow as tf

cur_folder = dirname(abspath(__file__))
sys.path.insert(0, '{}/../../src'.format(cur_folder))

from utils import sorted_nicely, get_ts


def solve_parent_dir():
    pass


def check_flags():
    if FLAGS.node_feat_name:
        assert (FLAGS.node_feat_encoder == 'onehot')
    else:
        assert ('constant_' in FLAGS.node_feat_encoder)
    assert (0 < FLAGS.valid_percentage < 1)
    assert (FLAGS.layer_num >= 1)
    assert (FLAGS.batch_size >= 1)
    assert (FLAGS.iters >= 0)
    assert (FLAGS.gpu >= -1)
    d = FLAGS.flag_values_dict()
    ln = d['layer_num']
    ls = [False] * ln
    for k in d.keys():
        if 'layer_' in k and 'gc' not in k and 'branch' not in k and 'id' not in k:
            lt = k.split('_')[1]
            if lt != 'num':
                i = int(lt) - 1
                if not (0 <= i < len(ls)):
                    raise RuntimeError('Wrong spec {}'.format(k))
                ls[i] = True
    for i, x in enumerate(ls):
        if not x:
            raise RuntimeError('layer {} not specified'.format(i + 1))
    if is_transductive():
        assert (FLAGS.layer_num == 1)  # can only have one layer
        assert (FLAGS.gemb_dim >= 1)
        assert (not FLAGS.dataset_super_large)


def get_flags(k, check=False):
    if hasattr(FLAGS, k):
        return getattr(FLAGS, k)
    else:
        if check:
            raise RuntimeError('Need flag {} which does not exist'.format(k))
        return None


def extract_config_code():
    with open(join(get_siamese_dir(), 'config.py')) as f:
        return f.read()


def convert_msec_to_sec_str(sec):
    return '{:.2f}msec'.format(sec * 1000)


def convert_long_time_to_str(sec):
    day = sec // (24 * 3600)
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    minutes = sec // 60
    sec %= 60
    seconds = sec
    return '{} days {} hours {} mins {:.1f} secs'.format(
        int(day), int(hour), int(minutes), seconds)


def get_siamese_dir():
    return cur_folder


def get_coarsen_level():
    if FLAGS.coarsening:
        return int(FLAGS.coarsening[6:])
    else:
        return 1


def is_transductive():
    return 'transductive' in FLAGS.model


def get_model_info_as_str(model_info_table=None):
    rtn = []
    d = FLAGS.flag_values_dict()
    for k in sorted_nicely(d.keys()):
        v = d[k]
        s = '{0:26} : {1}'.format(k, v)
        rtn.append(s)
        if model_info_table:
            model_info_table.append([k, '**{}**'.format(v)])
    rtn.append('{0:26} : {1}'.format('ts', get_ts()))
    return '\n'.join(rtn)




def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


def truth_is_dist_sim():
    if FLAGS.ds_metric == 'ged':
        sim_or_dist = 'dist'
    else:
        assert (FLAGS.ds_metric == 'mcs')
        sim_or_dist = 'sim'
    return sim_or_dist


def reset_flag(func, str, v):
    delattr(FLAGS, str)
    func(str, v, '')


def clear_all_flags():
    for k in FLAGS.flag_values_dict():
        delattr(FLAGS, k)
