"""
utils functions:
"""

from torch_geometric.data import InMemoryDataset,Data
from torch_geometric.io import read_txt_array
import torch
import os.path as osp
import numpy as np
from torch_scatter import scatter_add
import torch_geometric
import torch.nn.functional as F

class LoadDataset(InMemoryDataset):
    """
    data loader for graph data
    """
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        super().__init__(root,transform,pre_transform,pre_filter)
        self.name=name
        self.root=root

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["docs.txt","edgelist.txt","labels.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        edge_path = osp.join(self.raw_dir,'{}.edgelist.txt'.format(self.name))
        edge_index = read_txt_array(edge_path, sep=',', dtype = torch.long).t()

        docs_path = osp.join(self.raw_dir, '{}.docs.txt'.format(self.name))
        f = open(docs_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding = 'UTF-8')
            content_list.append(line)
        x = np.array(content_list, dtype = int)
        x = torch.from_numpy(x).to(torch.float)

        label_path = osp.join(self.raw_dir,'{}.labels.txt'.format(self.name))
        f = open(label_path,'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding = 'UTF-8')
            line = line.replace("\r","").replace("\n","")
            content_list.append(line)
        y = np.array(content_list,dtype = int)
        y = torch.from_numpy(y).to(torch.int64)

        data_list = []
        data = Data(x = x, edge_index = edge_index,y = y )
        random_node_indices = np.random.permutation(y.shape[0])
        training_size = int(len(random_node_indices) * 0.8)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:training_size]
        val_node_indices = random_node_indices[training_size:training_size+val_size]
        test_node_indices = random_node_indices[training_size+val_size:]

        train_masks = torch.zeros([y.shape[0]], dtype = torch.bool)
        train_masks[train_node_indices] = 1
        val_masks = torch.zeros([y.shape[0]], dtype = torch.bool)
        val_masks[val_node_indices] = 1
        test_masks = torch.zeros([y.shape[0]], dtype = torch.bool)
        test_masks[test_node_indices] = 1

        data.train_mask = train_masks
        data.val_mask = val_masks
        data.test_mask = test_masks

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list.append(data)

        data, slices = self.collate([data])

        torch.save((data, slices), self.processed_paths[0])


def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    """
    make long-tailed data by removing nodes sequentially
    method comes form paper 'GraphENS: Neighbor-Aware Ego Network Synthesis for Class-Imbalanced Node Classification'
    code:https://github.com/JoonHyung-Park/GraphENS
    """
    # sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    #compute number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu,i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu,i), sorted_n_data[i])))
        """
        remove low degree nodes sequentially(10 steps)
        since degrees of remaining nodes are changed when some nodes are moved
        """
        #we do not remove the nodes of magjor
        if i <1 :
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    #compute the removing nodes for each class
    remove_class_num_list = [n_data[i].item() - class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()

    device = label.device
    index_list = index_list.to(device)

    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])
    # compute remove nodes index
    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # find already removed nodes
            node_mask = label.new_ones(label.size(), dtype = torch.bool)
            node_mask[sum(remove_idx_list, [])] = False

            # remove connection with removed nodes in edge
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size = label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # remove low degree nodes first
            _, remove_idx = torch.topk(degree, (r * remove_class_num_list[i])//n_round[i], largest = False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())

    # find removed nodes
    node_mask = label.new_ones(label.size(), dtype = torch.bool)
    node_mask[sum(remove_idx_list, [])] = False

    # remove connection with removed nodes in edge
    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def sampling_anchor_nodes(data, k, use_ppmi = True):
    """
    sampling anchor nodes for each class according to degrees

    parameters:
    k: number of anchor nodes for each class
    use_ppmi: sampling for local or global branch
    return: a list of anchor nodes
    """
    labels = data.y
    anchor_nodes = []
    label_class = torch.unique(labels)

    # sampling for global branch
    if use_ppmi:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.global_adj[0], num_nodes = data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())

    # sampling for local branch
    else:
        for label in label_class:
            class_nodes = (labels == label).nonzero().squeeze()
            degrees = torch_geometric.utils.degree(data.edge_index[0], num_nodes = data.num_nodes)
            _, sorted_indices = torch.sort(degrees[class_nodes], descending=True)
            anchor_nodes.extend(class_nodes[sorted_indices[:k]].tolist())

    return anchor_nodes


def calculate_prototypes(anchor_nodes, node_embeddings, labels):
    """
    calculate prototypes for each class
    return: (tensor) prototypes
    """
    label_class = torch.unique(labels)
    prototypes = []

    for label in label_class:
        class_nodes = (labels == label).nonzero().squeeze()
        class_nodes_embeddings = node_embeddings[class_nodes]
        prototype = torch.mean(class_nodes_embeddings, dim = 0)
        prototypes.append(prototype.unsqueeze(0))
    prototypes = torch.cat(prototypes, dim=0)
    return prototypes


def cross_branch_prototype_contrastive_loss(local_anchor_nodes,global_anchor_nodes, local_nodes_embeddings,global_nodes_embeddings, labels, counts,dynamic_temperature=0.5):
    """
    calculate cross branch CL loss
    """

    local_prototypes = calculate_prototypes(local_anchor_nodes, local_nodes_embeddings, labels)
    global_prototypes = calculate_prototypes(global_anchor_nodes, global_nodes_embeddings, labels)

    # normalization
    local_prototypes = F.normalize(local_prototypes, dim=1)
    local_nodes_embeddings = F.normalize(local_nodes_embeddings, dim=1)
    global_prototypes = F.normalize(global_prototypes, dim=1)
    global_nodes_embeddings = F.normalize(global_nodes_embeddings, dim=1)

    prototypes = [local_prototypes,global_prototypes]
    nodes_embeddings = [local_nodes_embeddings,global_nodes_embeddings]
    anchor_nodes = [local_anchor_nodes,global_anchor_nodes]
    max_count = counts.max()
    label_class = torch.unique(labels)
    loss = 0

    for i in range(len(prototypes)):
        branch_prototype = prototypes[i]

        for label in label_class:
            count = counts[label]
            # calculate dynamic temperature for each class
            temp = dynamic_temperature + (1 - dynamic_temperature) * (count / max_count)
            prototype = branch_prototype[label]
            # cross branch
            for j in range(len(anchor_nodes)):
                class_nodes = [node for node in anchor_nodes[j] if labels[node] == label]
                branch_similarity = torch.exp(torch.matmul(nodes_embeddings[j],prototype.unsqueeze(1)).t() / temp)
                sum_branch_similarity = torch.sum(branch_similarity,dim=1)
                class_similarity = torch.exp(torch.matmul(nodes_embeddings[j][class_nodes],prototype.unsqueeze(1)).t() / temp)
                loss += -torch.mean(torch.log(class_similarity / sum_branch_similarity))
    N=((len(prototypes)+len(nodes_embeddings))*(label_class.max().item()+1))
    return loss/N


def cross_domain_contrastive_loss(local_anchor_nodes,global_anchor_nodes,local_nodes_embeddings,global_nodes_embeddings,target_embedding,labels,temperature=1):
    """
    calculate cross domain CL loss
    """
    anchor_nodes = [local_anchor_nodes,global_anchor_nodes]
    source_embedding=[local_nodes_embeddings,global_nodes_embeddings]
    loss = 0.0

    for i in range(len(anchor_nodes)):

        target_embedding_nor = F.normalize(target_embedding,dim=1)
        branch_source_embedding_nor=F.normalize(source_embedding[i][anchor_nodes[i]],dim=1)

        # calculate pseudo label
        similarities = torch.exp(torch.matmul(target_embedding_nor,branch_source_embedding_nor.T) / temperature)
        similarities_sums=similarities.sum(dim=1,keepdim=True)
        normalized_similarities = similarities / similarities_sums
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels[anchor_nodes[i]], num_classes=num_classes).float().to(labels.device)
        pseudo_labels = torch.matmul(normalized_similarities, one_hot_labels)
        target_pseudo_labels=pseudo_labels.argmax(dim=1)

        for class_idx in range(num_classes):
            target_class_nodes = (target_pseudo_labels == class_idx).nonzero().squeeze(1)
            if target_class_nodes.size(0) == 0:
                continue
            target_class_embedding = target_embedding[target_class_nodes]

            source_class_nodes=[node for node in anchor_nodes[i] if labels[node] == class_idx]
            source_class_embedding = source_embedding[i][source_class_nodes]

            target_class_embedding_nor = F.normalize(target_class_embedding,dim=1)
            source_class_embedding_nor=F.normalize(source_class_embedding,dim=1)

            similarity = torch.exp(torch.matmul(target_class_embedding_nor, source_class_embedding_nor.T) / temperature)
            similarity_sums = similarity.sum(dim=1,keepdim=True)
            nor_similarity = similarity / similarity_sums
            target_loss = -torch.log(nor_similarity).mean(dim=1)
            loss += target_loss.mean()/target_loss.size(0)
    return loss / len(anchor_nodes)