import torch
from torch import nn
from torch_geometric.nn.conv import RGCNConv
from torch_geometric.nn import GAT
from torch_geometric.utils import to_dense_adj
from torch.nn import functional as F
import math


class RGCN(nn.Module):
    def __init__(self, num_relations, hidden_dim, out_channels=1, num_layers=3, device='cpu'):
        super(RGCN, self).__init__()
        self.device = device
        self.embedding_dim = hidden_dim
        self.num_layers = num_layers

        self.conv = []

        # Define R-GCN layers
        for layer in range(num_layers - 1):
            self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=hidden_dim, num_relations=num_relations, num_bases=2).to(self.device))
        
        self.conv.append(RGCNConv(in_channels=hidden_dim, out_channels=out_channels, num_relations=num_relations, num_bases=2).to(self.device))
    
    def forward(self, x, edge_index, edge_type):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        
        # R-GCN layers
        for layer in range(self.num_layers):
            x = self.conv[layer](x, edge_index, edge_type)
        return x

class DiffusionOrderingNetwork(nn.Module):
    '''
    at each diffusion step t, we sample from this network to select a node 
    v_sigma(t) to be absorbed and obtain the corresponding masked graph Gt
    '''
    def __init__(self,
                 node_feature_dim,
                 num_node_types,
                 num_edge_types,
                 num_layers=3,
                 out_channels=1,
                 hidden_dim=32,
                 num_heads=6,
                 device='cpu'):
        super(DiffusionOrderingNetwork, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels

        num_node_types += 1 # add one for masked node type
        num_edge_types += 2 # add one for masked edge type and one for empty edge type
        
        # add positional encodings into node features
        self.embedding = nn.Embedding(num_embeddings=num_node_types, embedding_dim=hidden_dim).to(self.device)
        
        # Create an instance of the RGCN model
        self.gat = RGCN(num_relations=num_edge_types,
                        hidden_dim=self.hidden_dim,
                        out_channels=self.out_channels,
                        num_layers=num_layers,
                        device=device).to(self.device)
        
        # initialize positional encodings
        MAX_NODES = 10000
        self.pe = self.positionalencoding(MAX_NODES).to(self.device)

        self.effective_size_weight = nn.Parameter(torch.tensor(1.0, device=self.device))

    
    def compute_effective_size(self, G, alpha=0.7):
        """
        Compute effective size for each node in a directed graph.
        For directed graphs, effective size has two components:
        - In-neighbor effective size = |n_in| - |tie_in| / |n_in|
        - Out-neighbor effective size = |n_out| - |tie_out| / |n_out|
    
        Returns combined effective size (sum of in and out components).
        """
        num_nodes = G.x.shape[0]
        effective_sizes = torch.zeros(num_nodes, device=self.device)
    
        # Convert to dense adjacency matrix (directed)
        adj = to_dense_adj(G.edge_index, max_num_nodes=num_nodes).squeeze(0)
    
        for i in range(num_nodes):
            # Get in-neighbors (nodes pointing TO node i)
            in_neighbors = torch.where(adj[:, i] > 0)[0]  # adj[:, i] means edges TO i
            n_in = len(in_neighbors)
        
            # Get out-neighbors (nodes pointing FROM node i)
            out_neighbors = torch.where(adj[i, :] > 0)[0]  # adj[i, :] means edges FROM i
            n_out = len(out_neighbors)
        
            # Compute in-neighbor effective size
            if n_in > 0:
                # Count edges among in-neighbors (ties)
                in_neighbor_adj = adj[in_neighbors][:, in_neighbors]
                in_neighbor_x = G.x[in_neighbors]

                if in_neighbor_x.dim() > 1:
                    in_neighbor_x = in_neighbor_x.squeeze(-1)

                if in_neighbor_x.dim() == 0:
                    in_neighbor_x = in_neighbor_x.unsqueeze(0)
                
                role_matrix = (in_neighbor_x.unsqueeze(0) == in_neighbor_x.unsqueeze(1))
                role_matrix.fill_diagonal_(False)
                tie_mask = (in_neighbor_adj > 0) & role_matrix
                tie_in = tie_mask.sum().item()

                eff_size_in = n_in - (tie_in / n_in)
            else:
                eff_size_in = 0.0
        
            # Compute out-neighbor effective size
            if n_out > 0:
                # Count edges among out-neighbors (ties)
                out_neighbor_adj = adj[out_neighbors][:, out_neighbors]
                out_neighbor_x = G.x[out_neighbors]

                if out_neighbor_x.dim() > 1:
                    out_neighbor_x = out_neighbor_x.squeeze(-1)
                
                if out_neighbor_x.dim() == 0:
                    out_neighbor_x = out_neighbor_x.unsqueeze(0)

                role_matrix = (out_neighbor_x.unsqueeze(0) == out_neighbor_x.unsqueeze(1))
                role_matrix.fill_diagonal_(False)
                tie_mask = (out_neighbor_adj > 0) & role_matrix
                tie_out = tie_mask.sum().item()
                eff_size_out = n_out - (tie_out / n_out)
            else:
                eff_size_out = 0.0
        
            # Combine in and out effective sizes (sum)
            effective_sizes_combined = eff_size_in * (1 - alpha) + eff_size_out * alpha

            max_possible = num_nodes - 1
            if max_possible > 0:
                effective_sizes[i] = effective_sizes_combined / max_possible
            else:
                effective_sizes[i] = 0.0
    
        return effective_sizes


    def positionalencoding(self, lengths):
        '''
        From Chen, et al. 2021 (Order Matters: Probabilistic Modeling of Node Sequences for Graph Generation)
        * lengths: length(s) of graph in the batch
        '''
        l_t = lengths # .max() # use when parallelizing
        pes = torch.zeros([l_t, self.out_channels], device=self.device)
        position = torch.arange(0, l_t, device=self.device).unsqueeze(1) + 1
        div_term = torch.exp((torch.arange(0, self.out_channels, 2, dtype=torch.float, device=self.device) *
                              -(math.log(10000.0) / self.out_channels)))
        pes[:,0::2] = torch.sin(position.float() * div_term)
        pes[:,1::2] = torch.cos(position.float() * div_term)
        return pes

    def forward(self, G, node_order=None):
        '''
        node_order: list of absorbed nodes so far
        '''
        # list of not absorbed nodes (G.x.shape[0], except for nodes in node_order)
        unmasked = torch.tensor([node for node in range(G.x.shape[0]) if node not in node_order], device=self.device)

        h = self.embedding(G.x.squeeze().long().to(self.device))

        # # Positional encoding
        for t in range(len(node_order)):
            h[node_order[t], :] += self.pe[t, :].to(self.device)
        h = self.gat(h, G.edge_index.long().to(self.device), G.edge_attr.long().to(self.device))

        if unmasked.numel() > 0:
            effective_sizes = self.compute_effective_size(G)
            h = h + self.effective_size_weight * effective_sizes.unsqueeze(1)
            h_unmasked = h[unmasked, :]

            # softmax: h over h_not_absorbed
            # make sure values are positive and sum to 1 (for unmasked nodes)
            h = torch.exp(h) / torch.sum(torch.exp(h_unmasked), dim=0)
            h[node_order, :] *= 0 # zero the probability for already absorbed nodes
        else:
            effective_sizes = self.compute_effective_size(G)
            h = h + self.effective_size_weight * effective_sizes.unsqueeze(1)

            h = torch.exp(h) / torch.sum(torch.exp(h), dim=0)
        
        return h  # outputs probabilities for a categorical distribution over nodes
