import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from torch_geometric.data import Data



class GATModel(nn.Module):
    """
    Node backbone = GAT; two heads:
      - node_out: (dx, dy) per node, tanh-bounded
      - edge_out: d_area per *directed* edge from [h_src || h_dst || edge_attr]

    NOTE: edge_attr is now used directly in the edge head.  # [NEW]
    """
    def __init__(
        self,
        node_in_features: int = 2,
        edge_in_features: int = 2,
        node_out_features: int = 2,
        edge_out_features: int = 1,
        hidden_dim: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_in_features, hidden_dim)

        self.gat1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim, hidden_dim // 1, heads=1, concat=True, dropout=dropout)

        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, node_out_features),
            nn.Tanh(),  # keep deltas bounded
        )

        self.edge_out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_features, hidden_dim),  # [NEW]
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, edge_out_features),
            nn.Tanh(),  # keep deltas bounded
        )

        self.dropout = dropout

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.node_encoder(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = nn.functional.elu(self.gat1(x, edge_index))
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)

        src = x[edge_index[0]]  # [E_dir, H]
        dst = x[edge_index[1]]  # [E_dir, H]
        e_in = torch.cat([src, dst, edge_attr], dim=-1)  # [E_dir, 2H + edge_in]  # [NEW]

        node_pred = self.node_out(x)       # [N, 2]
        edge_pred = self.edge_out_mlp(e_in)  # [E_dir, 1]  # [NEW]
        edge_pred = edge_pred.squeeze(-1)    # [E_dir]

        return edge_pred, node_pred


class GCNModel(nn.Module):
    """GCN backbone with the same node/edge heads as GAT."""
    def __init__(
        self,
        node_in_features: int = 2,
        edge_in_features: int = 2,
        node_out_features: int = 2,
        edge_out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        self.enc = nn.Linear(node_in_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(dropout)

        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, node_out_features),
            nn.Tanh(),
        )

        self.edge_out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_features, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, edge_out_features),
            nn.Tanh(),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.enc(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = self.drop(x)

        src, dst = x[edge_index[0]], x[edge_index[1]]
        e_in = torch.cat([src, dst, edge_attr], dim=-1)

        n_out = self.node_out(x)
        e_out = self.edge_out_mlp(e_in).squeeze(-1)
        return e_out, n_out


class SAGEModel(nn.Module):
    """GraphSAGE backbone with the same node/edge heads as GAT/GCN."""
    def __init__(
        self,
        node_in_features: int = 2,
        edge_in_features: int = 2,
        node_out_features: int = 2,
        edge_out_features: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.enc = nn.Linear(node_in_features, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.LeakyReLU(0.01)
        self.drop = nn.Dropout(dropout)

        self.node_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, node_out_features),
            nn.Tanh(),
        )

        self.edge_out_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_features, hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden_dim, edge_out_features),
            nn.Tanh(),
        )

    def forward(self, data: Data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.enc(x)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = self.act(x)
            x = self.drop(x)

        src, dst = x[edge_index[0]], x[edge_index[1]]
        e_in = torch.cat([src, dst, edge_attr], dim=-1)

        n_out = self.node_out(x)
        e_out = self.edge_out_mlp(e_in).squeeze(-1)
        return e_out, n_out
