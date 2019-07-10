from utils import get_root_path
import json
import networkx as nx


def normalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    return 2 * d / (g1_size + g2_size)


def unnormalized_dist_sim(d, g1, g2, dec_gsize=False):
    g1_size = g1.number_of_nodes()
    g2_size = g2.number_of_nodes()
    if dec_gsize:
        g1_size -= 1
        g2_size -= 1
    return d * (g1_size + g2_size) / 2


def map_node_type_to_float(g):
    for n, attr in g.nodes(data=True):
        if 'type' in attr:
            s = ''
            for c in attr['type']:
                num = ord(c)
                s = str(num)
            num = int(s)
            attr['hed_mapped'] = num
        else:
            attr['hed_mapped'] = 0
    for n1, n2, attr in g.edges(data=True):
        attr['hed_mapped'] = 0
    return g


def _get_label_map(g1, g2, label_key):
    # Need this function because the two graphs needs consistent labelings in the mivia format. If they are called
    # separately, then they will likely have wrong labelings.
    label_dict = {}
    label_counter = 0
    # We make the labels into ints so that they can fit in the 16 bytes needed
    # for the labels in the mivia format. Each unique label encountered just gets a
    # unique label from 0 to edge_num - 1
    for g in [g1, g2]:
        for node, attr in g.nodes(data=True):
            current_label = attr[label_key]
            if current_label not in label_dict:
                label_dict[current_label] = label_counter
                label_counter += 1
    return label_dict


def mcis_edge_map_from_nodes(g1, g2, node_mapping):
    edge_map = {}
    induced_g1 = g1.subgraph([str(key) for key in node_mapping.keys()])
    induced_g2 = g2.subgraph([str(key) for key in node_mapping.values()])

    used_edge_ids_g2 = set()
    for u1, v1, edge1_attr in induced_g1.edges(data=True):
        u2 = str(node_mapping[int(u1)])
        v2 = str(node_mapping[int(v1)])
        edge1_id = edge1_attr['id']
        for temp1, temp2, edge2_attr in induced_g2.edges_iter(nbunch=[u2, v2], data=True):
            if (u2 == temp1 and v2 == temp2) or (u2 == temp2 and v2 == temp1):
                edge2_id = edge2_attr['id']
                if edge2_id in used_edge_ids_g2:
                    continue
                used_edge_ids_g2.add(edge2_id)
                edge_map[edge1_id] = edge2_id

    return edge_map


def write_mivia_input_file(graph, filepath, labeled, label_key, label_map):
    bytes, idx_to_node = convert_to_mivia(graph, labeled, label_key, label_map)
    with open(filepath, 'wb') as writefile:
        for byte in bytes:
            writefile.write(byte)
    return idx_to_node


def get_mcs_info_cpp(g1, g2, mivia_edge_mapping):
    g1_edge_map = get_mivia_edge_map(g1)
    g2_edge_map = get_mivia_edge_map(g2)

    # Translate the mivia edge map to nx edge id map.
    edge_map = {}
    for mivia_edge_1, mivia_edge_2 in mivia_edge_mapping.items():
        edge_1 = g1_edge_map[mivia_edge_1]
        edge_2 = g2_edge_map[mivia_edge_2]
        edge_map[edge_1] = edge_2

    mcs_node_id_maps, mcs_node_label_maps = get_mcs_info(g1, g2, [edge_map])

    return mcs_node_id_maps, mcs_node_label_maps, [edge_map]


def node_id_map_to_label_map(g1, g2, node_id_map):
    node_label_map = {}
    for (source1, target1), (source2, target2) in node_id_map.items():
        g1_edge = (g1.node[source1]['label'], g1.node[target1]['label'])
        g2_edge = (g2.node[source2]['label'], g2.node[target2]['label'])
        node_label_map[g1_edge] = g2_edge
    return node_label_map


def get_node_id_map_from_edge_map(id_edge_map1, id_edge_map2, edge_mapping):
    node_map = {}
    for edge1, edge2 in edge_mapping.items():
        nodes_edge1 = id_edge_map1[edge1]
        nodes_edge2 = id_edge_map2[edge2]
        nodes1 = (nodes_edge1[0], nodes_edge1[1])
        nodes2 = (nodes_edge2[0], nodes_edge2[1])
        node_map[nodes1] = nodes2
    return node_map


def get_id_edge_map(graph):
    id_edge_map = {}
    for u, v, edge_data in graph.edges(data=True):
        edge_id = edge_data['id']
        assert edge_id not in id_edge_map
        id_edge_map[edge_id] = (u, v)
    return id_edge_map


def get_mcs_path():
    return get_root_path() + '/model/mcs'


def write_java_input_file(g1, g2, algo, filepath):
    """Prepares and writes a file in JSON format for MCS calculation."""
    write_data = {}
    write_data['graph1'] = graph_as_dict(g1)
    write_data['graph2'] = graph_as_dict(g2)
    write_data['algorithm'] = algo
    # Assume there's at least one node and get its attributes
    test_node_attr = g1.nodes_iter(data=True).__next__()[1]
    # This is the actual key we want the MCS algorithm to use to compare node labels. The
    # Java MCS code has a default "unlabeled" key, so for unlabeled graphs, can just use that.
    write_data['nodeLabelKey'] = 'type' if 'type' in test_node_attr else 'unlabeled'

    with open(filepath, 'w') as jsonfile:
        json.dump(write_data, jsonfile)


def graph_as_dict(graph):
    dict = {}
    dict['directed'] = nx.is_directed(graph)
    dict['gid'] = graph.graph['gid']
    dict['nodes'] = []
    dict['edges'] = []
    for node, attr in graph.nodes(data=True):
        node_data = {}
        node_data['id'] = node
        node_data['label'] = attr['label']
        if 'type' in attr:
            node_data['type'] = attr['type']
        dict['nodes'].append(node_data)
    for source, target, attr in graph.edges(data=True):
        dict['edges'].append({'id': attr['id'], 'source': source, 'target': target})
    return dict
