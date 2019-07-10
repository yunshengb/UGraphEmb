import tensorflow as tf

# Hyper-parameters.
flags = tf.app.flags

# For data preprocessing.
dataset = 'ptc'
dataset_train = dataset
dataset_val_test = dataset
if 'aids' in dataset or dataset in ['webeasy', 'nci109', 'ptc']:
    node_feat_name = 'type'
    node_feat_encoder = 'onehot'
    max_nodes = 10
    num_glabels = 2
    if dataset == 'webeasy':
        max_nodes = 404
        num_glabels = 20
    if dataset == 'nci109':
        max_nodes = 106
    if dataset == 'ptc':
        max_nodes = 109
elif 'imdb' in dataset:
    node_feat_name = None
    node_feat_encoder = 'constant_1'
    max_nodes = 90
    num_glabels = 3
else:
    assert (False)
flags.DEFINE_string('dataset_train', dataset_train, 'Dataset for training.')
flags.DEFINE_string('dataset_val_test', dataset_val_test, 'Dataset for testing.')
flags.DEFINE_integer('num_glabels', num_glabels, 'Number of graph labels in the dataset.')
flags.DEFINE_string('node_feat_name', node_feat_name, 'Name of the node feature.')
flags.DEFINE_string('node_feat_encoder', node_feat_encoder,
                    'How to encode the node feature.')
""" valid_percentage: (0, 1). """
flags.DEFINE_float('valid_percentage', 0.25,
                   '(# validation graphs) / (# validation + # training graphs.')
ds_metric = 'ged'
flags.DEFINE_string('ds_metric', ds_metric, 'Distance/Similarity metric to use.')
if ds_metric == 'ged':
    ds_algo = 'astar'
else:
    raise NotImplementedError()
flags.DEFINE_string('ds_algo', ds_algo,
                    'Ground-truth distance algorithm to use.')
""" ordering: 'bfs', 'degree', None. """
flags.DEFINE_string('ordering', 'bfs', '')
""" coarsening: 'metis_<num_level>', None. """
flags.DEFINE_string('coarsening', None, 'Algorithm for graph coarsening.')
flags.DEFINE_string('laplacian', 'gcn', '')

# For model.
model = 'siamese_regression'
flags.DEFINE_string('model', model, 'Model string.')
flags.DEFINE_string('model_name', 'Our Model', 'Model name string.')
flags.DEFINE_integer('batch_size', 2, 'Number of graph pairs in a batch.')
ds_norm = True
flags.DEFINE_boolean('ds_norm', ds_norm,
                     'Whether to normalize the distance or not '
                     'when choosing the ground truth distance.')
flags.DEFINE_boolean('node_embs_norm', False,
                     'Whether to normalize the node embeddings or not.')
pred_sim_dist, supply_sim_dist = None, None
if model == 'siamese_regression':
    """ ds_kernel: gaussian, exp, inverse, identity. """
    ds_kernel = 'exp'
    if ds_metric == 'glet':  # already a sim metric
        ds_kernel = 'identity'
    flags.DEFINE_string('ds_kernel', ds_kernel,
                        'Name of the similarity kernel.')
    if ds_kernel == 'gaussian':
        """ yeta:
         if ds_norm, try 0.6 for nef small, 0.3 for nef, 0.2 for regular;
         else, try 0.01 for nef, 0.001 for regular. """
        flags.DEFINE_float('yeta', 0.01, 'yeta for the gaussian kernel function.')
    elif ds_kernel == 'exp' or ds_kernel == 'inverse':
        flags.DEFINE_float('scale', 0.7, 'Scale for the exp/inverse kernel function.')
    pred_sim_dist = 'sim'  # check!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ds_metric == 'mcs' or ds_metric == 'glet':
        pred_sim_dist = 'sim'
    supply_sim_dist = pred_sim_dist
    # Start of mse loss.
    lambda_msel = 1  # 1  # 1 #0.0001
    if lambda_msel > 0:
        flags.DEFINE_float('lambda_mse_loss', lambda_msel,
                           'Lambda for the mse loss.')
    # End of mse loss.
    # Start of weighted distance loss.
    lambda_wdl = 0  # 1#1  # 1
    if lambda_wdl > 0:
        flags.DEFINE_float('lambda_weighted_dist_loss', lambda_wdl,
                           'Lambda for the weighted distance loss.')
        supply_sim_dist = 'sim'  # special for wdl loss
        # flags.DEFINE_boolean('graph_embs_norm', True,
        #                      'Whether to normalize the graph embeddings or not.')
    # End of weighted distance loss.
    # Start of trivial solution avoidance loss.
    lambda_tsl = 0
    if lambda_tsl > 0:
        flags.DEFINE_float('lambda_triv_avoid_loss', lambda_tsl,
                           'Lambda for the trivial solution avoidance loss.')
    # End of trivial solution avoidance loss.
    # Start of diversity encouraging loss.
    lambda_del = 0
    if lambda_del > 0:
        flags.DEFINE_float('lambda_diversity_loss', lambda_del,
                           'Lambda for the diversity encouraging loss.')
    # End of diversity encouraging loss.
flags.DEFINE_string('pred_sim_dist', pred_sim_dist,
                    'dist/sim indicating whether the model is predicting dist or sim.')
flags.DEFINE_string('supply_sim_dist', supply_sim_dist,
                    'dist/sim indicating whether the model should supply dist or sim.')
layer = 0
layer += 1
if model == 'siamese_regression':

    # '''
    # --------------------------------- ATT+NTN ---------------------------------
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:output_dim=256,dropout=False,bias=True,'
        'act=relu,sparse_inputs=True,type=gcn', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Coarsening:pool_style=max', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:input_dim=256,output_dim=128,dropout=False,bias=True,'
        'act=relu,sparse_inputs=False,type=gcn', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Coarsening:pool_style=max', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'GraphConvolution:input_dim=128,output_dim=64,dropout=False,bias=True,'
        'act=identity,sparse_inputs=False,type=gcn', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GraphConvolution:input_dim=64,output_dim=64,dropout=False,bias=True,'
    #     'act=relu,sparse_inputs=False', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GraphConvolution:input_dim=64,output_dim=64,dropout=False,bias=True,'
    #     'act=identity,sparse_inputs=False', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Coarsening:pool_style=max', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'GMNAverage:input_dim=32,output_dim=128', '')  # Supersource, Average, GMNAverage
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Average', '')  # Supersource, Average, GMNAverage
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Attention:input_dim=64,att_times=1,att_num=1,att_weight=True,att_style=dot', '')
    gcn_num = layer
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'JumpingKnowledge:gcn_num={},gcn_layer_ids=1_2_3,'
        'input_dims=256_128_64,att_times=1,att_num=1,att_weight=True,att_style=dot,'
        'combine_method=concat'.format(gcn_num), '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'NTN:input_dim=64,feature_map_dim=64,dropout=False,bias=True,'
    #     'inneract=relu,apply_u=False', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'ANPM:input_dim=32,att_times=1,att_num=1,att_style=dot,att_weight=True,'
    #     'feature_map_dim=32,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=identity,mne_method=hist_32,branch_style=anpm', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'ANPM:input_dim=32,att_times=1,att_num=1,att_style=dot,att_weight=True,'
    #     'feature_map_dim=32,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=identity,mne_method=arg_max_naive,branch_style=pm', '')
    # flags.DEFINE_string(
    #     'layer_4',
    #     'ANPMD:input_dim=16,att_times=1,att_num=1,att_style=dot,att_weight=True,'
    #     'feature_map_dim=16,dropout=False,bias=True,'
    #     'ntn_inneract=relu,apply_u=False,'
    #     'padding_value=0,'
    #     'mne_inneract=sigmoid,mne_method=hist_16,branch_style=anpm,'
    #     'dense1_dropout=False,dense1_act=relu,dense1_bias=True,dense1_output_dim=8,'
    #     'dense2_dropout=False,dense2_act=relu,dense2_bias=True,dense2_output_dim=4', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=64,output_dim=32,dropout=False,bias=True,'
    #     'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=32,output_dim=16,dropout=False,bias=True,'
    #     'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=16,output_dim=8,dropout=False,bias=True,'
    #     'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=8,output_dim=4,dropout=False,bias=True,'
    #     'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=4,output_dim=1,dropout=False,bias=True,'
    #     'act=identity', '')

    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=1,output_dim=1,dropout=False,bias=True,'
    #     'act=identity', '')

    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=448,output_dim=348,dropout=False,bias=True,'
        'act=relu', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=348,output_dim=256,dropout=False,bias=True,'
        'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=64,output_dim=128,dropout=False,bias=True,'
    #     'act=relu', '')
    # layer += 1
    # flags.DEFINE_string(
    #     'layer_{}'.format(layer),
    #     'Dense:input_dim=128,output_dim=128,dropout=False,bias=True,'
    #     'act=identity', '')
    layer += 1
    flags.DEFINE_string(
        'layer_{}'.format(layer),
        'Dense:input_dim=256,output_dim=256,dropout=False,bias=True,'
        'act=identity', '')
    flags.DEFINE_integer('gemb_layer_id', layer, 'Layer index (1-based) '
                                                 'to obgtain graph embeddings.')
    layer += 1
    if flags.FLAGS.pred_sim_dist == 'dist':
        flags.DEFINE_string(
            'layer_{}'.format(layer), 'Dist:norm=None', '')
    else:
        flags.DEFINE_string(
            'layer_{}'.format(layer), 'Dot:output_dim=1,act=identity', '')
    flags.DEFINE_integer('layer_num', layer, 'Number of layers.')
# '''
# Start of graph loss.
""" graph_loss: '1st', None. """
graph_loss = None
flags.DEFINE_string('graph_loss', graph_loss, 'Loss function(s) to use.')
if graph_loss:
    flags.DEFINE_float('graph_loss_alpha', 0.,
                       'Weight parameter for the graph loss function.')
# End of graph loss.

flags.DEFINE_float('train_real_percent', 1, '')

# Supersource node.
# Referenced as "super node" in https://arxiv.org/pdf/1511.05493.pdf.
# Node that is connected to all other nodes in the graph.
flags.DEFINE_boolean('supersource', False,
                     'Boolean. Whether or not to use a supersouce node in all of the graphs.')
# Random walk generation and usage.
# As used in the GraphSAGE model implementation: https://github.com/williamleif/GraphSAGE.
flags.DEFINE_string('random_walk', None,
                    'Random walk configuration. Set none to not use random walks. Format is: '
                    '<num_walks>_<walk_length>')

# Training (optimiztion) details.
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')
""" learning_rate: 0.01 recommended. """
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

# For training and validating.
flags.DEFINE_integer('gpu', -1, 'Which gpu to use.')  # -1: cpu
flags.DEFINE_integer('iters', 10, 'Number of iterations to train.')

# For testing.
flags.DEFINE_boolean('plot_results', True,
                     'Whether to plot the results '
                     '(involving all baselines) or not.')
flags.DEFINE_integer('plot_max_num', 10, 'Max number of plots per experiment.')
flags.DEFINE_integer('max_nodes', max_nodes, 'Maximum number of nodes in a graph.')

FLAGS = tf.app.flags.FLAGS
