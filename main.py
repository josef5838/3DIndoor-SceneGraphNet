from utils.default_settings import *
from utils.utl import try_mkdir
import argparse
from SceneGraphNet.train import train_model

''' parser input '''
parser = argparse.ArgumentParser()

# train process settings
parser.add_argument('--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--reg_lr', type=float, default=1e-5, help='weight decay')
parser.add_argument('--d_vec_dim', type=int, default=100, help='feature dimension for encoded vector')
parser.add_argument('--h_vec_dim', type=int, default=300, help='feature dimension for hidden layers')

# model variants
parser.add_argument('--K', type=int, default=3, help='times of iteration')
parser.add_argument('--aggregate_in_order', default=True, action='store_false', help='if aggregating object features in distance order')
parser.add_argument('--aggregation_func', default='GRU', help='aggregation function, choice=[GRU, CatRNN, MaxPool, Sum]')
parser.add_argument('--cat_msg', default=False, action='store_true', help='if true, use MLP to predict message passing, else, directly use node representation as message')

# room type settings
parser.add_argument('--room_type', type=str, default='3RScan', help='room type, choice=[bedroom, living, bathroom, office, 3RScan]')
parser.add_argument('--num_train_rooms', default=1000, type=int, help='number of rooms for training')
parser.add_argument('--num_test_rooms', default=100, type=int, help='number of rooms for testing')

# for load and test on pretrained model
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--load_model_name', type=str, default='', help='dir of pretrained model')
parser.add_argument('--load_model_along_with_optimizer', default=False, action='store_true', help='if load pretrained model along with optimizer')

# others
parser.add_argument('--verbose', default=0, type=int, help='')
parser.add_argument('--name', default='my-train-model')


opt_parser = parser.parse_args()
opt_parser.write = not opt_parser.test

id2cat_file = open('data/preprocess/TRAIN_id2cat_{}.json'.format(opt_parser.room_type))
opt_parser.id2cat = json.load(id2cat_file)
opt_parser.cat2id = {opt_parser.id2cat[id]: id for id in opt_parser.id2cat.keys()}
rels = ["None"]
with open("data/relationships.txt") as rfile:
    for line in rfile:
            # Remove leading/trailing whitespace
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            # Replace spaces with underscores and add to the list
            rel = line
            rels.append(rel)
opt_parser.rels = rels
rel2id = {}
for idx, rel in enumerate(rels):
    rel2id[rel]=idx
# import pdb; pdb.set_trace()
opt_parser.rel2id = rel2id
if(opt_parser.load_model_name != ''):
    opt_parser.ckpt = os.path.join(ckpt_dir, opt_parser.load_model_name, 'Entire_model_max_acc.pth')
else:
    opt_parser.ckpt = ''

opt_parser.outf = os.path.join(ckpt_dir, opt_parser.name)
try_mkdir(opt_parser.outf)

M = train_model(opt_parser=opt_parser)

if(not opt_parser.test):
    for epoch in range(opt_parser.nepoch):
        M.train(epoch)
        M.test(epoch)
else:
    M.test(0)
