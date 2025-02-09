import args
import os.path as osp
from utils import *
from torch_geometric.data import Data
import pickle
from sklearn.metrics import f1_score
from Models import *
import random
import numpy as np
import torch
import itertools

def encode(x, edge_index, cache_name, encoder, use_ppmi = True, compensate = False, mask = None):
    x = x.to(args.device)
    edge_index = edge_index.to(args.device)
    encoded_output = encoder(x, edge_index, cache_name,compensate, args.device, use_ppmi = use_ppmi)
    if mask is not None:
        encoded_output = encoded_output[mask]
    return encoded_output


def predict(data, cache_name, classifier_model, encoder, compensate, mask = None):
    encoded_output = encode(data.x, data.edge_index, cache_name, encoder)
    logits = classifier_model(encoded_output)
    return logits


def evaluate(preds, labels):
    f1_micro = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='micro')
    f1_macro = f1_score(labels.cpu().detach(), preds.cpu().detach(), average='macro')
    return f1_micro, f1_macro


def test(data, cache_name, models,compensate = False, mask = None):
    for model in models:
        model.eval()
    encoder = models[0]
    classifier_model = models[1]
    logits = predict(data, cache_name, classifier_model, encoder,compensate, mask)
    preds = logits.argmax(dim=1)
    labels = data.y if mask is None else data.y[mask]
    f1_micro, f1_macro = evaluate(preds, labels)
    return f1_micro, f1_macro


def train(models, optimizer, source_data_train, target_data_train):
    for model in models:
        model.train()
    encoder = models[0]
    classifier_model = models[1]

    local_source_embedding = encode(source_data_train.x, source_data_train.edge_idnex, imbalance_cache_name, encoder, use_ppmi=False, compensate=True)
    local_target_embedding = encode(target_data_train.x, target_data_train.edge_idnex, args.target, encoder, use_ppmi=False)

    global_source_embedding = encode(source_data_train.x, source_data_train.edge_index, imbalance_cache_name, encoder, use_ppmi=True, compensate=True)
    global_target_embedding = encode(target_data_train.x, target_data_train.edge_idnex, args.target, encoder, use_ppmi=True)

    source_embedding = local_source_embedding + global_source_embedding
    target_embedding = local_target_embedding + global_target_embedding

    local_anchor_nodes = sampling_anchor_nodes(source_data_train, args.k, use_ppmi=False)
    global_anchor_nodes = sampling_anchor_nodes(source_data_train, args.k, use_ppmi=True)
    labels = source_data_train.y

    # calculate cross branch prototype CL
    _, counts = label.unique(return_counts=True)
    cb_cL_loss = cross_branch_prototype_contrastive_loss(local_anchor_nodes, global_anchor_nodes, local_source_embedding, global_source_embedding, labels, counts, args.dynamic_temperature)

    # calculate cross domain CL
    cd_cl_loss = cross_domain_contrastive_loss(local_anchor_nodes, global_anchor_nodes, local_source_embedding, global_source_embedding, target_embedding, labels)

    logits = classifier_model(global_source_embedding)
    lc_loss = loss_func(logits, labels)

    loss = cb_cL_loss + cd_cl_loss + lc_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


'''----------------------begin_training----------------------------------'''
args = args.getargs()
print(args)

#prepare data
source_path = osp.join(osp.dirname(osp.realpath(__file__)),'./','data',args.source)
source_dataset = LoadDataset(source_path, args.source)

target_path = osp.join(osp.dirname(osp.realpath(__file__)),'./','data',args.target)
target_dataset = LoadDataset(target_path, args.target)
source_data = source_dataset[0].to(args.device)
target_data = target_dataset[0].to(args.device)
imbalance_cache_name = args.source + str(args.imbalance_factor)

#make long_tail data
train_mask = source_data.train_mask.clone()
label = source_data.y
n_cls = label.max().item() + 1
n_data = []
train_label = label[train_mask]
for i in range(n_cls):
    data_num = (train_label == i).sum()
    n_data.append(int(data_num.item()))

class_num_list, new_train_mask, idx_info, node_mask, edge_mask =\
    make_longtailed_data_remove(source_data.edge_index, label, n_data, n_cls, args.imbalance_factor, train_mask.clone())

x = source_data.x
y = source_data.y
edge_index = source_data.edge_index

x_train = x[new_train_mask]
y_train = y[new_train_mask]

# re-mapping edge_index
train_node_indices = torch.nonzero(new_train_mask).squeeze()
node_map = {original_idx.item(): new_idx for new_idx, original_idx in enumerate(train_node_indices)}
new_edge_index = edge_index[:, edge_mask]
new_edge_index[0] = torch.tensor([node_map.get(idx.item(), -1) for idx in new_edge_index[0]], device = edge_index.device)
new_edge_index[1] = torch.tensor([node_map.get(idx.item(), -1) for idx in new_edge_index[1]], device = edge_index.device)
valid_edge_mask = (new_edge_index[0] != -1) & (new_edge_index[1] != -1)
new_edge_index = new_edge_index[:, valid_edge_mask]

imbalance_source_data = Data(
    x = x_train,
    y = y_train,
    edge_index = new_edge_index,
).to(args.device)

with open('temp/'+imbalance_cache_name+'.pkl', 'rb') as f:
    source_edge_index, norm= pickle.load(f)
with open('temp/'+args.target+'.pkl', 'rb') as f:
    target_edge_index, norm = pickle.load(f)

imbalance_source_data.global_adj = source_edge_index.to(args.device)
target_data.global_adj = target_edge_index.to(args.device)
loss_func = nn.CrossEntropyLoss().to(args.device)

print("number of nodes for each class:", class_num_list)
print("Load data done! Start experiment!")


def main():
    best_f1_micro_per_time=[]
    best_f1_macro_per_time=[]


    for time in range(args.times):
        print("begin experiment {}".format(time+1))
        # set seed
        random.seed(time)
        np.random.seed(time)
        torch.manual_seed(time)
        torch.cuda.manual_seed(time)

        # set models
        encoder = GNNEncoder(imbalance_source_data, args.encoder_dim, args.dropout, source_data.x.shape[1], args.device).to(args.device)
        classifier_model = nn.Sequential(nn.Linear(args.encoder_dim, n_cls)).to(args.device)
        models = [encoder, classifier_model]
        params = itertools.chain(*[model.parameters() for model in models])
        optimizer = torch.optim.Adam(params, lr = args.learning_rate, weight_decay=args.weight_decay)

        # begin a training with 200 epochs
        best_f1_micro = 0.0
        best_f1_macro = 0.0
        for epoch in range(args.epochs):
            train(models,optimizer,imbalance_source_data,target_data)
            f1_micro, f1_macro = test(target_data, args.target,models)
            if epoch % 10 == 0:
                print("time: {}, Epoch: {}, f1_micro: {:.4f}, f1_macro: {:.4f}".format(time+1, epoch,  f1_micro,f1_macro))
            if f1_micro > best_f1_micro:
                best_f1_micro = f1_micro
                best_f1_macro = f1_macro
        print("============================================================")
        line = " experiment {} done, results: best_f1_micro: {:.4f}, best_f1_macro: {:.4f}" \
            .format( time+1, best_f1_micro, best_f1_macro)
        print(line)
        best_f1_micro_per_time.append(best_f1_micro)
        best_f1_macro_per_time.append(best_f1_macro)

    best_f1_micro_per_time_np=np.array(best_f1_micro_per_time)
    best_f1_macro_per_time_np=np.array(best_f1_macro_per_time)
    micro_avg_value = np.mean(best_f1_micro_per_time_np)
    micro_std_value = np.std(best_f1_micro_per_time_np)
    macro_avg_value = np.mean(best_f1_macro_per_time_np)
    macro_std_value = np.std(best_f1_macro_per_time_np)
    print("===============================================================================================\n")
    print("source:{}, target:{}, imbalanced factor: {}, experiment time: {},task done!".format(args.source,args.target,args.imbalance_factor,args.times))

    print("best f1_micro  results:")
    for value in best_f1_micro_per_time:
        print("{:.4f}".format(value))

    print("best macro  results:")
    for value in best_f1_macro_per_time:
        print("{:.4f}".format(value))

    print("micro average value:",micro_avg_value)
    print("micro std value:",micro_std_value)
    print("macro average value:",macro_avg_value)
    print("macro std value:",macro_std_value)


if __name__ == "__main__":
    main()




