import sys, os
# from utils.txt_to_SUNCS import txt2SUNCS, convert_dataset
from utils.default_settings import *
import argparse
import torch
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.data
from utils.utl import weight_init
import copy
import SceneGraphNet.model as model
from SceneGraphNet.model import to_torch, device
import torchfold
import time
from tqdm import tqdm
import json

    

disable_progress = not sys.stdout.isatty()

class inference_model():
    def __init__(self, opt_parser):
        """
        Initialize training process
        :param opt_parser: args parser
        """

        ''' Input args parser '''
        self.opt_parser = opt_parser
        self.model = model

        ''' Initialize model '''
        # Haoliang
        k_size = len(opt_parser.cat2id.keys()) + 2
        
        self.rels = opt_parser.rels
        
        # encoder
        self.full_enc = self.model.FullEnc(rels=self.rels, k=k_size, d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim)
        self.full_enc.apply(weight_init)
        self.full_enc.to(device)

        #Haoliang

        self.full_dec = self.model.RelationshipDec(d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim, num_rels=len(opt_parser.rel2id))
        self.full_dec.apply(weight_init)
        self.full_dec.to(device)
        

        # ''' Setup Optimizer '''
        # self.opt = {}
        # self.opt['full_enc'] = optim.Adam(self.full_enc.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)
        # self.opt['full_dec'] = optim.Adam(self.full_dec.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

        ''' Load pre-trained model '''
        self.pretrained_epoch = 0
        if opt_parser.ckpt != '':
            ckpt = torch.load(opt_parser.ckpt, map_location=device)

            def update_partial_dict(model, pretrained_ckpt):
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_ckpt.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                return model_dict

            self.full_enc.load_state_dict(update_partial_dict(self.full_enc, ckpt['full_enc_state_dict']))
            self.full_dec.load_state_dict(update_partial_dict(self.full_dec, ckpt['full_dec_state_dict']))
            # if(opt_parser.load_model_along_with_optimizer):
            #     self.opt['full_enc'].load_state_dict(ckpt['full_enc_opt'])
            #     self.opt['full_dec'].load_state_dict(ckpt['full_dec_opt'])
            self.pretrained_epoch = ckpt['epoch']
            print("=========== LOAD PRE TRAINED MODEL ENCODER: " + opt_parser.ckpt + " ==================")
            print(f"pretrained epoch: {self.pretrained_epoch}")
    def _add_node(self, node_list, node_name):
        # add a new node to node_list
        if node_name not in node_list:
            node_list[node_name] = {}
            node_list[node_name]["type"] = node_name.rsplit("_", 1)[0]
            node_list[node_name]["self_info"] = {}
            for rel in self.rels:
                node_list[node_name][rel] = []
        return node_list
    def _get_entropy(self, logits):
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        return entropy
    def _keep_rel_by_entropy(self, rel_preds, entropies, topk=2):
        # keep topk relationship by entropy
        entropies = torch.Tensor(entropies)
        rel_preds = rel_preds.clone()
        sorted_idx = torch.argsort(entropies)
        if topk < len(sorted_idx):
            rel_preds[sorted_idx[topk:]] = -1
        return rel_preds

    def _entropy_filtering(self, logits, threshold):
        entropy = self._get_entropy(logits)
        return entropy.item() < threshold
    def _inference_pass(self, valid_rooms, task):
        """
        Single training pass
        :param valid_rooms: choice of =[self.valid_rooms_train, self.valid_rooms_test]
        :param epoch: current epoch
        :param is_training: train or test pass
        :return:
        """

        ''' epoch and args '''
        opt_parser =self.opt_parser
        topk = 3
        ''' current training state '''
        self.full_enc.eval()
        self.full_dec.eval()

        all_ranks = []
        
        if task == 'mani':
            room_indices = list(range(len(valid_rooms)))
            room_idx_batches = [room_indices[i: i + opt_parser.batch_size] for i in
                            range(0, len(valid_rooms), opt_parser.batch_size)]
            ''' Batch loop '''
            for batch_i, batch in enumerate(tqdm(room_idx_batches, desc="Batches")):

                batch_rooms = [valid_rooms[i] for i in batch]

                """ ==================================================================
                                            Encoder Part
                ================================================================== """

                # loop for rooms
                for room_i, room in enumerate(tqdm(batch_rooms, desc="Rooms", leave=False)):
                    
                    original_node_list = room['node_list']
                    
                    # augment the dataset such that for relationships that have opposite relationships, set the opposite relationship for the flipped pair
                    opp_rels = {
                        "higher_than": "lower_than",
                        "lower_than": "higher_than",
                        "left": "right",
                        "right": "left",
                        "front": "behind",
                        "behind": "front",
                        "bigger_than": "smaller_than",
                        "smaller_than": "bigger_than",
                        "same_as": "same_as"
                    }
                    for rel in opp_rels:
                        if rel in self.rels and opp_rels[rel] in self.rels:
                            for node_name, node_info in original_node_list.items():
                                if isinstance(node_info, dict):
                                    if rel in node_info:
                                        for i, val in enumerate(node_info[rel]):
                                            if val in original_node_list:
                                                if opp_rels[rel] not in node_info:
                                                    node_info[opp_rels[rel]] = []
                                                if node_name not in original_node_list[val][opp_rels[rel]]:
                                                    original_node_list[val][opp_rels[rel]].append(node_name)
                    # import pdb; pdb.set_trace()
                                                        

                    # Process each node as a query node.
                    for query_node in list(original_node_list.keys()):
                        # Deep copy the node list for a reduced graph.
                        reduced_node_list = copy.deepcopy(original_node_list)

                        # Additionally, remove any references to the query node from all relationship lists.
                        # Here, we assume that the relationship keys are provided in self.rels.
                        for node_name, node_info in reduced_node_list.items():
                            for rel in self.rels:
                                if rel in node_info:
                                    # Assume the relationship value is a list of node names.
                                    if isinstance(node_info[rel], list):
                                        node_info[rel] = [n for n in node_info[rel] if n != query_node]
                                    # If it's a single value instead, clear it if it matches.
                                    elif node_info[rel] == query_node:
                                        node_info[rel] = None

                        for rel in self.rels:
                            if rel == "None":
                                continue
                            # remove all other relationship exept the one we are interested in
                            rel_node_list = reduced_node_list[query_node][rel]
                            if isinstance(rel_node_list, list):
                                if len(rel_node_list) == 0:
                                    continue
                                for target_node in rel_node_list:
                                    # first recover the original target node list
                                    reduced_node_list[query_node][rel] = original_node_list[query_node][rel]
                                    if target_node in original_node_list:
                                        reduced_node_list[query_node][rel] = [target_node]
                                        # This rel is the only one that should be present in the reduced graph.
                                        # --- Message Passing for this specific edge begins here ---
                                        enc_fold = torchfold.Fold()
                                        # Build the fold operations and get the list of node handles
                                        encoded_nodes = self.model.encode_graph_fold(enc_fold, reduced_node_list, opt_parser)
                                        # Use the returned list of handles when applying the fold
                                        # Wrap each node handle in a list so that each is iterable.
                                        node_handles = [[node] for node in encoded_nodes.values()]
                                        enc_fold_nodes = enc_fold.apply(self.full_enc, node_handles)

                                        # Create a mapping to decouple the node names from the encoded nodes.
                                        node_names = list(encoded_nodes.keys())
                                        enc_mapping = {}
                                        for i, name in enumerate(node_names):
                                            # Assuming enc_fold_results is a list or can be indexed accordingly.
                                            enc_mapping[name] = enc_fold_nodes[i]

                                        dec_fold = torchfold.Fold()
                                        dec_fold_nodes = []
                                        # Get the query node's encoded vector.
                                        query_d_vec = enc_mapping[query_node]

                                        # For every remaining node in the reduced graph, add a decoder op.
                                        query_gt_labels = []
                                        for known_node in node_names:
                                            if known_node == query_node or known_node == target_node:
                                                continue
                                            # target node uses gt relationship
                                            # Look up the ground-truth relationship from the original room.
                                            gt_rel = next((rel for rel, values in original_node_list[query_node].items() 
                                                        if isinstance(values, list) and known_node in values), "None")
                                            if gt_rel == "None":
                                                continue
                                            node_handle = dec_fold.add('full_dec', query_d_vec, enc_mapping[known_node])
                                            dec_fold_nodes.append(node_handle)
                                            gt = opt_parser.rel2id[gt_rel]
                                            query_gt_labels.append(gt)

                                        # If no (query, known) pairs were found, skip this query.
                                        if len(query_gt_labels) == 0:
                                            continue

                                        # Execute the fold to obtain relationship logits.
                                        dec_fold_nodes_wrapped = [[node] for node in dec_fold_nodes]

                                        # Now, call apply with the wrapped nodes.
                                        fold_out = dec_fold.apply(self.full_dec, dec_fold_nodes_wrapped)

                                        logits = torch.cat(fold_out, dim=0)


                                        # Create ground-truth tensor.
                                        gt_tensor = to_torch(query_gt_labels, torch_type=torch.LongTensor, dim_0=len(query_gt_labels)).view(-1)

                                        sorted_logits, sorted_indices = logits.sort(dim=1, descending=True)

                                        # Compute the rank of the true label for each candidate.

                                        for i in range(gt_tensor.size(0)):
                                            # Find the position where the ground-truth label appears.
                                            # Adding 1 since we want ranks starting from 1.
                                            rank = (sorted_indices[i] == gt_tensor[i]).nonzero(as_tuple=True)[0].item() + 1
                                            all_ranks.append(rank)
        if all_ranks:
            all_ranks_tensor = torch.tensor(all_ranks, dtype=torch.float)
            mean_rank = all_ranks_tensor.mean().item()              # Mean Rank (MR)
            mrr = (1.0 / all_ranks_tensor).mean().item()              # Mean Reciprocal Rank (MRR)
            hits1 = (all_ranks_tensor <= 1).float().mean().item()       # Hits@1
            hits3 = (all_ranks_tensor <= 3).float().mean().item()       # Hits@3
            hits10 = (all_ranks_tensor <= 10).float().mean().item()     # Hits@10

            print(f"Aggregated Metrics across all valid_rooms:")
            print(f"MR: {mean_rank:.4f}, MRR: {mrr:.4f}, Hits@1: {hits1:.4f}, Hits@3: {hits3:.4f}, Hits@10: {hits10:.4f}")
        else:
            print("No valid queries were processed for metric computation.")
    def inference(self):
        st = time.time()
        self._inference_pass(self.opt_parser.valid_room, self.opt_parser.task)
        print('time usage:', time.time() - st)

if __name__ == "__main__":

    
    
    
    opt_parser = argparse.Namespace()
    
    opt_parser.task = 'mani'
    opt_parser.room_type = '3RScan'
    id2cat_file_path = os.path.join(root_dir, 'data/preprocess/TRAIN_id2cat_{}.json'.format(opt_parser.room_type))
    with open(id2cat_file_path) as id2cat_file:
        opt_parser.id2cat = json.load(id2cat_file)
    opt_parser.cat2id = {opt_parser.id2cat[id]: id for id in opt_parser.id2cat.keys()}
    rels = ["None"]
    with open(os.path.join(root_dir, "data/relationships.txt")) as rfile:
        for line in rfile:
                # Remove leading/trailing whitespace
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                # Replace spaces with underscores and add to the list
                rel = line
                rels.append(rel)
    dataset_path = os.path.join(root_dir, 'data/3RScan_validation.json')
    with open(dataset_path) as dataset_file:
        dataset = json.load(dataset_file)
    valid_rooms = dataset#convert_dataset(dataset, rel_types=rels)
    opt_parser.valid_room = valid_rooms
    opt_parser.rels = rels
    rel2id = {}
    for idx, rel in enumerate(rels):
        rel2id[rel]=idx
    id2rel = {v: k for k, v in rel2id.items()}
    opt_parser.id2rel = id2rel
    opt_parser.rel2id = rel2id
    opt_parser.ckpt = os.path.join(ckpt_dir, opt_parser.task, "min_loss.pth")
    # opt_parser.outf = os.path.join(scene_path, f"{opt_parser.task}_sgnet.txt")
    # try_mkdir(opt_parser.outf)

    opt_parser.nepoch = 100
    opt_parser.batch_size = 4
    opt_parser.lr = 5e-4
    opt_parser.reg_lr = 1e-5
    opt_parser.d_vec_dim = 100
    opt_parser.h_vec_dim = 300
    opt_parser.K = 3

    opt_parser.cat_msg = False

    M = inference_model(opt_parser=opt_parser)
    M.inference()

    
    

