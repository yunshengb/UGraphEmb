from config import FLAGS
from layers import GraphConvolution, GraphConvolutionAttention, Coarsening, Average, \
    Attention, Dot, Dist, NTN, SLM, Dense, Padding, \
    GraphConvolutionCollector, PadandTruncate, Supersource, GMNAverage, \
    JumpingKnowledge
import tensorflow as tf
import numpy as np
from math import exp


def create_layers(model, pattern, num_layers):
    layers = []
    for i in range(1, num_layers + 1):  # 1-indexed
        sp = FLAGS.flag_values_dict()['{}_{}'.format(pattern, i)].split(':')
        name = sp[0]
        layer_info = {}
        if len(sp) > 1:
            assert (len(sp) == 2)
            for spec in sp[1].split(','):
                ssp = spec.split('=')
                layer_info[ssp[0]] = ssp[1]
        if name == 'GraphConvolution':
            layers.append(create_GraphConvolution_layer(layer_info, model, i))
        elif name == 'GraphConvolutionAttention':
            layers.append(create_GraphConvolutionAttention_layer(layer_info, model, i))
        elif name == 'GraphConvolutionCollector':
            layers.append(create_GraphConvolutionCollector_layer(layer_info))
        elif name == 'Coarsening':
            layers.append(create_Coarsening_layer(layer_info))
        elif name == 'Average':
            layers.append(create_Average_layer(layer_info))
        elif name == 'Attention':
            layers.append(create_Attention_layer(layer_info))
        elif name == 'Supersource':
            layers.append(create_Supersource_layer(layer_info))
        elif name == 'GMNAverage':
            layers.append(create_GMNAverage_layer(layer_info))
        elif name == 'JumpingKnowledge':
            layers.append(create_JumpingKnowledge_layer(layer_info))
        elif name == 'Dot':
            layers.append(create_Dot_layer(layer_info))
        elif name == 'Dist':
            layers.append(create_Dist_layer(layer_info))
        elif name == 'SLM':
            layers.append(create_SLM_layer(layer_info))
        elif name == 'NTN':
            layers.append(create_NTN_layer(layer_info))
        elif name == 'Dense':
            layers.append(create_Dense_layer(layer_info))
        elif name == 'Padding':
            layers.append(create_Padding_layer(layer_info))
        elif name == 'PadandTruncate':
            layers.append(create_PadandTruncate_layer(layer_info))
        else:
            raise RuntimeError('Unknown layer {}'.format(name))
    return layers


def create_GraphConvolution_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 7:
        raise RuntimeError('GraphConvolution layer must have 5-7 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolution(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1,
        type=layer_info['type'] if 'type' in layer_info else 'gcn')


def create_GraphConvolutionAttention_layer(layer_info, model, layer_id):
    if not 5 <= len(layer_info) <= 6:
        raise RuntimeError('GraphConvolution layer must have 3-4 specs')
    input_dim = layer_info.get('input_dim')
    if not input_dim:
        if layer_id != 1:
            raise RuntimeError(
                'The input dim for layer must be specified'.format(layer_id))
        input_dim = model.input_dim
    else:
        input_dim = int(input_dim)
    return GraphConvolutionAttention(
        input_dim=input_dim,
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        sparse_inputs=parse_as_bool(layer_info['sparse_inputs']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']),
        featureless=False,
        num_supports=1)


def create_GraphConvolutionCollector_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('GraphConvolutionCollector layer must have 5 spec')
    return GraphConvolutionCollector(gcn_num=int(layer_info['gcn_num']),
                                     fix_size=int(layer_info['fix_size']),
                                     mode=int(layer_info['mode']),
                                     padding_value=int(layer_info['padding_value']),
                                     align_corners=parse_as_bool(layer_info['align_corners']))


def create_Coarsening_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Coarsening layer must have 1 spec')
    return Coarsening(pool_style=layer_info['pool_style'])


def create_Average_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Average layer must have 0 specs')
    return Average()


def create_Attention_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Attention layer must have 5 specs')
    return Attention(input_dim=int(layer_info['input_dim']),
                     att_times=int(layer_info['att_times']),
                     att_num=int(layer_info['att_num']),
                     att_style=layer_info['att_style'],
                     att_weight=parse_as_bool(layer_info['att_weight']))


def create_Supersource_layer(layer_info):
    if not len(layer_info) == 0:
        raise RuntimeError('Supersource layer must have 0 specs')
    return Supersource()


def create_GMNAverage_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('GMNAverage layer must have 2 specs')
    return GMNAverage(input_dim=int(layer_info['input_dim']),
                      output_dim=int(layer_info['output_dim']))


def create_JumpingKnowledge_layer(layer_info):
    if not len(layer_info) == 8:
        raise RuntimeError('JumpingKnowledge layer must have 8 specs')
    return JumpingKnowledge(gcn_num=int(layer_info['gcn_num']),
                            gcn_layer_ids=parse_as_int_list(
                                layer_info['gcn_layer_ids']),
                            input_dims=parse_as_int_list(layer_info['input_dims']),
                            att_times=int(layer_info['att_times']),
                            att_num=int(layer_info['att_num']),
                            att_style=layer_info['att_style'],
                            att_weight=parse_as_bool(layer_info['att_weight']),
                            combine_method=layer_info['combine_method'])


def create_Dot_layer(layer_info):
    if not len(layer_info) == 2:
        raise RuntimeError('Dot layer must have 2 specs')
    return Dot(output_dim=int(layer_info['output_dim']),
               act=create_activation(layer_info['act']))


def create_Dist_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Dot layer must have 1 specs')
    return Dist(norm=layer_info['norm'])


def create_SLM_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('SLM layer must have 5 specs')
    return SLM(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        act=create_activation(layer_info['act']),
        dropout=parse_as_bool(layer_info['dropout']),
        bias=parse_as_bool(layer_info['bias']))


def create_NTN_layer(layer_info):
    if not len(layer_info) == 6:
        raise RuntimeError('NTN layer must have 6 specs')
    return NTN(
        input_dim=int(layer_info['input_dim']),
        feature_map_dim=int(layer_info['feature_map_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        inneract=create_activation(layer_info['inneract']),
        apply_u=parse_as_bool(layer_info['apply_u']),
        bias=parse_as_bool(layer_info['bias']))


def create_Dense_layer(layer_info):
    if not len(layer_info) == 5:
        raise RuntimeError('Dense layer must have 5 specs')
    return Dense(
        input_dim=int(layer_info['input_dim']),
        output_dim=int(layer_info['output_dim']),
        dropout=parse_as_bool(layer_info['dropout']),
        act=create_activation(layer_info['act']),
        bias=parse_as_bool(layer_info['bias']))


def create_Padding_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('Padding layer must have 1 specs')
    return Padding(
        padding_value=int(layer_info['padding_value']))


def create_PadandTruncate_layer(layer_info):
    if not len(layer_info) == 1:
        raise RuntimeError('PadandTruncate layer must have 1 specs')
    return PadandTruncate(
        padding_value=int(layer_info['padding_value']))


def create_activation(act, ds_kernel=None, use_tf=True):
    if act == 'relu':
        return tf.nn.relu if use_tf else relu_np
    elif act == 'identity':
        return tf.identity if use_tf else identity_np
    elif act == 'sigmoid':
        return tf.sigmoid if use_tf else sigmoid_np
    elif act == 'tanh':
        return tf.tanh if use_tf else np.tanh
    elif act == 'ds_kernel':
        return ds_kernel.dist_to_sim_tf if use_tf else \
            ds_kernel.dist_to_sim_np
    else:
        raise RuntimeError('Unknown activation function {}'.format(act))


def relu_np(x):
    return np.maximum(x, 0)


def identity_np(x):
    return x


def sigmoid_np(x):
    try:
        ans = exp(-x)
    except OverflowError:  # TODO: fix
        ans = float('inf')
    return 1 / (1 + ans)


def parse_as_bool(b):
    if b == 'True':
        return True
    elif b == 'False':
        return False
    else:
        raise RuntimeError('Unknown bool string {}'.format(b))


def parse_as_int_list(il):
    rtn = []
    for x in il.split('_'):
        x = int(x)
        rtn.append(x)
    return rtn
