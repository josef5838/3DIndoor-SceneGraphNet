import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
# from utils.default_settings import dic_id2type
import copy
import numpy as np
import types

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

''' to_torch Variable '''
def to_torch(n, torch_type=torch.FloatTensor, requires_grad=False, dim_0=1):
    n = torch.tensor(n, requires_grad=requires_grad).type(torch_type).to(device)
    n = n.view(dim_0, -1)
    return n

def get_gt_k_vec(node_list, cur_node, opt_parser):
    """
    Get cur_node's k-vec = category
    :param node_list:
    :param cur_node:
    :param opt_parser:
    :return:
    """

    if (node_list[cur_node]['type'] == 'root'):
        cat = 'wall'
    elif (node_list[cur_node]['type'] == 'wall'):
        cat = 'wall'
    else:
        cat = node_list[cur_node]['type']

    cat_vec = [0.0] * (len(opt_parser.cat2id.keys()) + 1)
    cat_vec[int(opt_parser.cat2id[cat])] = 1.0

    return cat_vec 

class AggregateGRUEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(AggregateGRUEnc, self).__init__()

        self.w_x = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )
        self.w_h = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, d)
        )
        self.msg = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=[False]):
        if(cat_msg[0]):
            msg = self.msg(torch.cat((cur_d_vec, d_vec), dim=1))
        else:
            msg = d_vec

        ht = self.w_h(pre_vec) + self.w_x(msg) * w
        # ht = self.act(ht)
        return ht


class UpdateEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300, r=20):
        # r: relation types
        super(UpdateEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(d * r, h),
            nn.ReLU(),
            nn.Linear(h, d),
            # nn.Tanh()
        )

    def forward(self, *args):
        # *args: relations, num: r
        feat = torch.cat(args, dim=1)
        d_vec = self.enc(feat)
        return d_vec



class BoxEnc(nn.Module):
    def __init__(self, k=54, d=100, h=300):
        super(BoxEnc, self).__init__()

        self.enc = nn.Sequential(
            nn.Linear(k, h),
            nn.ReLU(),
            nn.Linear(h, d),
        )

    def forward(self, k_vec):
        d_vec = self.enc(k_vec)
        return d_vec


class FullEnc(nn.Module):
    def __init__(self, rels, k=55, d=100, h=300):
        super(FullEnc, self).__init__()

        for rel in rels:
            setattr(self, f'aggregate_{rel}_enc', AggregateGRUEnc(k, d, h))

            # Create the corresponding aggregate function
            func = self._make_aggregate_func(rel)
            # Bind the function as a method of this instance
            setattr(self, f'aggregate_{rel}_func', types.MethodType(func, self))


        # 27 relationship types
        self.aggregate_cooc_enc = AggregateGRUEnc(k, d, h)

        self.aggregate_self_enc = UpdateEnc(k, d, h)

        self.box_enc = BoxEnc(k, d, h)

    def _make_aggregate_func(self, rel):
        # This function returns a new function that calls the corresponding encoder
        def aggregate_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
            # Look up the corresponding encoder using the relation name
            enc = getattr(self, f'aggregate_{rel}_enc')
            return enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)
        return aggregate_func

    def aggregate_cooc_func(self, d_vec, pre_vec, cur_d_vec, w=1.0, cat_msg=False):
        return self.aggregate_cooc_enc(d_vec, pre_vec, cur_d_vec, w, cat_msg)

    def learned_weight_func(self, k_vec1, k_vec2, offset_vec):
        return self.learned_weight(k_vec1, k_vec2, offset_vec)


    def aggregate_self_func(self, *args):
        return self.aggregate_self_enc(*args)

    def cat_self_func(self, *args):
        return torch.cat((args), dim=1)

    def box_enc_func(self, k_vec):
        return self.box_enc(k_vec)


def encode_graph_fold(fold, raw_node_list, opt_parser):
    node_list = copy.deepcopy(raw_node_list)
    d_vec_dim = opt_parser.d_vec_dim
    encode_fold_list = {}

    def encode_node(node_list, step=0, rels=opt_parser.rels):
        """
        Graph message passing by torchfold encoding
        :param node_list:
        :param leaf_node:
        :param step:
        :return:
        """

        # init d-vec for all nodes
        if(step == 0):

            # loop to get each node's k-vec and d-vec
            for cur_node in node_list.keys():
                
                node_list[cur_node]['k-vec'] = get_gt_k_vec(node_list, cur_node, opt_parser)

                node_list[cur_node]['k-vec'] = to_torch(node_list[cur_node]['k-vec'])
                node_list[cur_node]['d-vec'] = fold.add('box_enc_func', node_list[cur_node]['k-vec'])
                
                    

        # graph message passing
        else:
            for cur_node in node_list.keys():
                cur_node_d_vec = node_list[cur_node]['pre-d-vec']

                # message from all links
                # Haoliang 
                aggregate_rel_d_vec = {}

                for rel in rels:
                    # Initialize aggregation for this relation
                    aggregate_rel_d_vec[rel] = to_torch([0.0] * d_vec_dim)
                    
                    if rel in node_list[cur_node]:
                        for neighbor_name in node_list[cur_node][rel]:
                            if neighbor_name in node_list:
                                neighbor_d_vec = node_list[neighbor_name]['pre-d-vec']
                                aggregate_rel_d_vec[rel] = fold.add(
                                    'aggregate_{}_func'.format(rel),
                                    neighbor_d_vec,
                                    aggregate_rel_d_vec[rel],
                                    cur_node_d_vec,
                                    to_torch([1.]),
                                    to_torch(opt_parser.cat_msg, torch.bool)
                                )

                # message from loose neighbors (default relation)
                aggregate_cooc_d_vec = to_torch([0.0] * d_vec_dim)
                all_neighbor_nodes = node_list.keys()
                for neighbor_node in all_neighbor_nodes:
                    if (neighbor_node != cur_node):
                        # w = node_list[cur_node]['w'][neighbor_node]
                        w = to_torch([1.])
                        neighbor_node_d_vec = node_list[neighbor_node]['pre-d-vec']
                        aggregate_cooc_d_vec = fold.add('aggregate_cooc_func',
                                                            neighbor_node_d_vec,
                                                            aggregate_cooc_d_vec,
                                                            cur_node_d_vec,
                                                            w,
                                                            to_torch([opt_parser.cat_msg], torch.bool))

                node_list[cur_node]['d-vec'] = fold.add('aggregate_self_func',
                                                        node_list[cur_node]['pre-d-vec'],
                                                        aggregate_rel_d_vec['supported_by'],
                                                        aggregate_rel_d_vec['left'],
                                                        aggregate_rel_d_vec['right'],
                                                        aggregate_rel_d_vec['front'],
                                                        aggregate_rel_d_vec['behind'],
                                                        aggregate_rel_d_vec['close_by'],
                                                        aggregate_rel_d_vec['bigger_than'],
                                                        aggregate_rel_d_vec['smaller_than'],
                                                        aggregate_rel_d_vec['higher_than'],
                                                        aggregate_rel_d_vec['lower_than'],
                                                        aggregate_rel_d_vec['same_as'],
                                                        aggregate_rel_d_vec['attached_to'],
                                                        aggregate_rel_d_vec['standing_on'],
                                                        aggregate_rel_d_vec['lying_on'],
                                                        aggregate_rel_d_vec['hanging_on'],
                                                        aggregate_rel_d_vec['connected_to'],
                                                        aggregate_rel_d_vec['leaning_against'],
                                                        aggregate_rel_d_vec['belonging_to'],
                                                        aggregate_cooc_d_vec)

            

        # end of func
    for i in range(opt_parser.K):
        encode_node(node_list, step=i)
        for cur_node in node_list.keys():
            node_list[cur_node]['pre-d-vec'] = node_list[cur_node]['d-vec']
    for cur_node in node_list.keys():
        encode_fold_list[cur_node] = node_list[cur_node]['d-vec']
    return encode_fold_list


# Haoliang
class RelationshipDec(nn.Module):
    def __init__(self, d, h, num_rels=27):
        """
        d: latent dimension (for both query and known node)
        h: hidden dimension in the decoder MLP
        num_rels: number of relationship types in your dataset (e.g., len(opt_parser.relationship_types))
        """
        super(RelationshipDec, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(d*2, h),
            nn.ReLU(),
            nn.Linear(h, num_rels)
        )
    def forward(self, query_d_vec, known_d_vec):
        # Expand query_d_vec to match known_d_vec if necessary.
        query_rep = query_d_vec.expand_as(known_d_vec)
        combined = torch.cat([query_rep, known_d_vec], dim=1)
        logits = self.dec(combined)
        return logits
    
    def full_dec(self, query_d_vec, known_d_vec):
        return self.forward(query_d_vec, known_d_vec)

