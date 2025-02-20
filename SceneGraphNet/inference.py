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
        k_size = len(opt_parser.cat2id.keys()) + 1
        
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
        if opt_parser.ckpt != '':
            ckpt = torch.load(opt_parser.ckpt, map_location=device)

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
            print(f"pretrained epoch: {self.pretrained_epoch}")

        ''' Loss function '''
        self.LOSS_CLS = torch.nn.CrossEntropyLoss()

        ''' load valid rooms '''
        with open(os.path.join(pkl_dir, '{}_data.json'.format(opt_parser.room_type))) as f:
            self.valid_rooms = json.load(f)
        
        self.valid_rooms_test = self.valid_rooms[11:12] # first a room for testing

        self.STATE = 'EVAL'
        self.MIN_LOSS = float('inf')

    def _inference_pass(self, valid_rooms, epoch):
        """
        Single training pass
        :param valid_rooms: choice of =[self.valid_rooms_train, self.valid_rooms_test]
        :param epoch: current epoch
        :param is_training: train or test pass
        :return:
        """

        ''' epoch and args '''
        opt_parser =self.opt_parser

        ''' current training state '''
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
        
        ''' Batch loop '''
        for batch_i, batch in enumerate(tqdm(room_idx_batches, desc="Batches")):

            batch_rooms = [valid_rooms[i] for i in batch]

            """ ==================================================================
                                        Encoder Part
            ================================================================== """

            # loop for rooms
            for room_i, room in enumerate(tqdm(batch_rooms, desc="Rooms", leave=False)):
                room_loss = 0.0
                room_pairs = 0
                
                original_node_list = room['node_list']

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

                    # Initialize a new torchfold fold for this query.
                     
                    enc_fold = torchfold.Fold()
                    # Build the fold operations and get the list of node handles
                    encoded_nodes = self.model.encode_graph_fold(enc_fold, reduced_node_list, opt_parser)
                    # Use the returned list of handles when applying the fold
                    # Wrap each node handle in a list so that each is iterable.
                    node_handles = [[node] for node in encoded_nodes.values()]
                    enc_fold_nodes = enc_fold.apply(self.full_enc, node_handles)

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
                    quadruplets = []
                    for idx, known_node in enumerate(node_names):
                        # Look up the ground-truth relationship from the original room.
                        gt_rel = next((rel for rel, values in original_node_list[query_node].items() 
                                    if isinstance(values, list) and known_node in values), "None")
                        if gt_rel == "None":
                            continue
                        node_handle = dec_fold.add('full_dec', query_d_vec, enc_mapping[known_node])
                        dec_fold_nodes.append(node_handle)
                        
                        
                        gt = opt_parser.rel2id[gt_rel]
                        query_gt_labels.append(gt)
                        quadruplets.append([query_node, known_node, gt_rel, idx])

                    # If no (query, known) pairs were found, skip this query.
                    if len(query_gt_labels) == 0:
                        continue

                    # Execute the fold to obtain relationship logits.
                    dec_fold_nodes_wrapped = [[node] for node in dec_fold_nodes]

                    # Now, call apply with the wrapped nodes.
                    fold_out = dec_fold.apply(self.full_dec, dec_fold_nodes_wrapped)

                    logits = torch.cat(fold_out, dim=0)

                    pred_rels = torch.argmax(logits, dim=1)

                    outputs = []
                    for idx, pred in enumerate(pred_rels):
                        # 0: query node, 1: known node, 2: ground-truth relationship, 3: predicted relationship
                        outputs.append([quadruplets[idx][0], quadruplets[idx][1], quadruplets[idx][2], opt_parser.id2rel[pred.item()]])

                    # add the rest of triplets
                    # for node in reduced_node_list:
                    #     for rel in self.rels:
                    #         if rel == "None":
                    #             continue
                    #         # import pdb; pdb.set_trace()
                    #         if isinstance(reduced_node_list[node][rel], list):
                    #             for obj in reduced_node_list[node][rel]:
                    #                 outputs.append([node, obj, rel, "No_Pred"])
                    
                    # save to txt
                    with open(os.path.join(opt_parser.outf, 'outputs_{}.txt'.format(epoch)), 'a') as f:
                        for output in outputs:
                            f.write(f"{output[0]} {output[1]} {output[2]} {output[3]}\n")


                    # Create ground-truth tensor.
                    gt_tensor = to_torch(query_gt_labels, torch_type=torch.LongTensor, dim_0=len(query_gt_labels)).view(-1)
                    # import pdb; pdb.set_trace()
                    loss = self.LOSS_CLS(logits, gt_tensor)

                    room_loss += loss.item() * len(query_gt_labels)
                    room_pairs += len(query_gt_labels)

                    total_loss += loss.item() * len(query_gt_labels)
                    total_examples += len(query_gt_labels)

                    msg = (f"{self.STATE} {opt_parser.name} Epoch {epoch}: "
                        f"Room {opt_parser.batch_size * batch_i + room_i}, Query {query_node} "
                        f"({len(query_gt_labels)} pairs) Loss: {loss.item():.4f}")
                    tqdm.write(msg)
            # if room_pairs > 0:
            #     avg_room_loss = room_loss / room_pairs
            #     tqdm.write(f"Room {opt_parser.batch_size * batch_i + room_i} Avg Loss: {avg_room_loss:.4f}")
        avg_loss = total_loss / total_examples if total_examples > 0 else 0
        tqdm.write("=" * 55)
        tqdm.write(f"{self.STATE} {epoch}: Avg Relationship Loss: {avg_loss:.4f}")
        tqdm.write("=" * 55)


    def inference(self, epoch):
        st = time.time()
        self._inference_pass(self.valid_rooms_test, epoch)
        print('time usage:', time.time() - st)