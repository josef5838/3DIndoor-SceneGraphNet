import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from utils.default_settings import *
from utils.utl import weight_init
import copy
import SceneGraphNet.model as model
from SceneGraphNet.model import to_torch, device
from random import shuffle
import torchfold
import time
from tqdm import tqdm
import random
import sys

# Disable progress bar if not running in a terminal (non-interactive)
disable_progress = not sys.stdout.isatty()


class train_model():
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
        # an extra dimension for the None relationship and uid
        
        self.rels = opt_parser.rels
        
        # encoder
        self.full_enc = self.model.FullEnc(rels=self.rels, k=k_size, d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim)
        self.full_enc.apply(weight_init)
        self.full_enc.to(device)

        #Haoliang

        self.full_dec = self.model.RelationshipDec(d=opt_parser.d_vec_dim, h=opt_parser.h_vec_dim, num_rels=len(opt_parser.rel2id))
        self.full_dec.apply(weight_init)
        self.full_dec.to(device)
        

        ''' Setup Optimizer '''
        self.opt = {}
        self.opt['full_enc'] = optim.Adam(self.full_enc.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)
        self.opt['full_dec'] = optim.Adam(self.full_dec.parameters(), lr=opt_parser.lr, weight_decay=opt_parser.reg_lr)

        ''' Load pre-trained model '''
        self.pretrained_epoch = 0
        if opt_parser.ckpt != '' and opt_parser.continue_training == True:
            ckpt = torch.load(opt_parser.ckpt)

            def update_partial_dict(model, pretrained_ckpt):
                model_dict = model.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_ckpt.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                return model_dict

            self.full_enc.load_state_dict(update_partial_dict(self.full_enc, ckpt['full_enc_state_dict']))
            self.full_dec.load_state_dict(update_partial_dict(self.full_dec, ckpt['full_dec_state_dict']))
            if(opt_parser.load_model_along_with_optimizer):
                self.opt['full_enc'].load_state_dict(ckpt['full_enc_opt'])
                self.opt['full_dec'].load_state_dict(ckpt['full_dec_opt'])
            self.pretrained_epoch = ckpt['epoch']
            print("=========== LOAD PRE TRAINED MODEL ENCODER: " + opt_parser.ckpt + " ==================")

        ''' Loss function '''
        self.LOSS_CLS = torch.nn.CrossEntropyLoss()

        ''' load valid rooms '''
        with open(os.path.join(pkl_dir, '{}_train.json'.format(opt_parser.room_type))) as f:
            self.valid_rooms = json.load(f)
        self.valid_rooms_train = self.valid_rooms[0:opt_parser.num_train_rooms]
        self.valid_rooms_test = self.valid_rooms[opt_parser.num_train_rooms:opt_parser.num_train_rooms + opt_parser.num_test_rooms]

        self.STATE = 'INIT'
        self.MIN_LOSS = float('inf')


   
    def _training_pass(self, valid_rooms, epoch, is_training=True, type='add'):
        """
        Single training pass
        :param valid_rooms: choice of =[self.valid_rooms_train, self.valid_rooms_test]
        :param epoch: current epoch
        :param is_training: train or test pass
        :return:
        """

        ''' epoch and args '''
        epoch += self.pretrained_epoch
        opt_parser =self.opt_parser

        ''' current training state '''
        if (is_training):
            self.STATE = 'TRAIN'
            self.full_enc.train()
            self.full_dec.train()
        else:
            self.STATE = 'EVAL'
            self.full_enc.eval()
            self.full_dec.eval()
    

        ''' init loss / accuracy '''
        total_loss, total_examples = 0.0, 0
        ''' shuffle room list and create training batches '''
        shuffle(valid_rooms)
        room_indices = list(range(len(valid_rooms)))
        room_idx_batches = [room_indices[i: i + opt_parser.batch_size] for i in
                            range(0, len(valid_rooms), opt_parser.batch_size)]
        
        if type == 'add':
            ''' Batch loop '''
            for batch_i, batch in enumerate(tqdm(room_idx_batches, desc="Batches", disable=disable_progress)):

                batch_rooms = [valid_rooms[i] for i in batch]

                """ ==================================================================
                                            Encoder Part
                ================================================================== """

                # loop for rooms
                for room_i, room in enumerate(tqdm(batch_rooms, desc="Rooms", leave=False, disable=disable_progress)):
                    room_loss = 0.0
                    room_pairs = 0
                    
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
                        # Remove the query node.
                        reduced_node_list.pop(query_node, None)

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
                        # For the query node, compute its representation solely from its category label.
                        query_k_vec = self.model.get_gt_k_vec(original_node_list, query_node, opt_parser)
                        query_k_vec = to_torch(query_k_vec)  # Convert to tensor.
                        query_d_vec = self.full_enc.box_enc_func(query_k_vec)

                        
                        # For every remaining node in the reduced graph, add a decoder op.
                        query_gt_labels = []
                        for known_node in node_names:
                            # Look up the ground-truth relationship from the original room.
                            gt_rel = next((rel for rel, values in original_node_list[query_node].items() 
                                        if isinstance(values, list) and known_node in values), "None")
                            if gt_rel == "None" and random.random() < 0.9:
                                continue
                            node_handle = dec_fold.add('full_dec', query_d_vec, enc_mapping[known_node])
                            dec_fold_nodes.append(node_handle)
                            
                            
                            gt = opt_parser.rel2id[gt_rel]
                            query_gt_labels.append(gt)
                        # compute loss weights as the inverse of the number of examples for each class
                        # loss_weights = torch.tensor([1.0 / len([x for x in query_gt_labels if x == i]) for i in range(len(opt_parser.rel2id))])
                        # loss_weights = loss_weights / loss_weights.sum()
                        # loss_weights = loss_weights.to(device)

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
                        # import pdb; pdb.set_trace()
                        loss = self.LOSS_CLS(logits, gt_tensor)

                        room_loss += loss.item() * len(query_gt_labels)
                        room_pairs += len(query_gt_labels)

                        total_loss += loss.item() * len(query_gt_labels)
                        total_examples += len(query_gt_labels)

                        if is_training:
                            for key in self.opt:
                                self.opt[key].zero_grad()
                            loss.backward()
                            for key in self.opt:
                                self.opt[key].step()

                        msg = (f"{self.STATE} {opt_parser.name} Epoch {epoch}: "
                            f"Room {opt_parser.batch_size * batch_i + room_i}, Query {query_node} "
                            f"({len(query_gt_labels)} pairs) Loss: {loss.item():.4f}")
                        tqdm.write(msg)
            # if room_pairs > 0:
            #     avg_room_loss = room_loss / room_pairs
            #     tqdm.write(f"Room {opt_parser.batch_size * batch_i + room_i} Avg Loss: {avg_room_loss:.4f}")
        elif type == 'mani':
            ''' Batch loop '''
            for batch_i, batch in enumerate(tqdm(room_idx_batches, desc="Batches", disable=disable_progress)):

                batch_rooms = [valid_rooms[i] for i in batch]

                """ ==================================================================
                                            Encoder Part
                ================================================================== """

                # loop for rooms
                for room_i, room in enumerate(tqdm(batch_rooms, desc="Rooms", leave=False, disable=disable_progress)):
                    room_loss = 0.0
                    room_pairs = 0
                    
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
                                            if gt_rel == "None" and random.random() < 0.9:
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
                                        # import pdb; pdb.set_trace()
                                        loss = self.LOSS_CLS(logits, gt_tensor)

                                        room_loss += loss.item() * len(query_gt_labels)
                                        room_pairs += len(query_gt_labels)

                                        total_loss += loss.item() * len(query_gt_labels)
                                        total_examples += len(query_gt_labels)

                                        if is_training:
                                            for key in self.opt:
                                                self.opt[key].zero_grad()
                                            loss.backward()
                                            for key in self.opt:
                                                self.opt[key].step()

                                        msg = (f"{self.STATE} {opt_parser.name} Epoch {epoch}: "
                                            f"Room {opt_parser.batch_size * batch_i + room_i}, Query {query_node} "
                                            f"({len(query_gt_labels)} pairs) Loss: {loss.item():.4f}")
                                        tqdm.write(msg)
        else:
            raise ValueError(f"Invalid type: {type}, please train for task add or mani")
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        tqdm.write("=" * 55)
        tqdm.write(f"{self.STATE} {epoch}: Avg Relationship Loss: {avg_loss:.4f}")
        tqdm.write("=" * 55)
        # Optionally save the model when evaluating.
        # if not is_training:
        def save_model(save_type):
            torch.save({
                'full_enc_state_dict': self.full_enc.state_dict(),
                'full_dec_state_dict': self.full_dec.state_dict(),
                'full_enc_opt': self.opt['full_enc'].state_dict(),
                'full_dec_opt': self.opt['full_dec'].state_dict(),
                'epoch': epoch
            }, f"{opt_parser.outf}/{save_type}.pth")
            tqdm.write(f"Saved model to {opt_parser.outf}/{save_type}.pth")

        if avg_loss < self.MIN_LOSS:
            self.MIN_LOSS = avg_loss
            save_model('min_loss')
        save_model(f'epoch_{epoch}')

        return

    def train(self, epoch):
        st = time.time()
        self._training_pass(self.valid_rooms_train, epoch, is_training=True, type=self.opt_parser.task)
        print('time usage:', time.time() - st)

    def test(self, epoch, DEBUG_mode=False):
        with torch.no_grad():
            self._training_pass(self.valid_rooms_test, epoch, is_training=False, type=self.opt_parser.task)



