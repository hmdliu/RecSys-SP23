
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMF(nn.Module):
    def __init__(self, config):
        super().__init__()
       
        # hyper-parameters
        self.config = config
        self.emb_dim = config['emb_dim']
        self.hidden_dims = config['hidden_dims']
        self.dropout_flag = config.get('dropout_flag', False)

        # GMF component
        self.gmf = GMF(config)

        # MLP component
        self.mlp = MLP(config)

        # NeuMF layer
        self.nmf = nn.Sequential(
            nn.Linear(self.emb_dim+self.hidden_dims[-1], 1),
            nn.Dropout(p=0.2) if self.dropout_flag else nn.Identity(),
            nn.Sigmoid()
        )
        
    def forward(self, user_idx, item_idx):
        x_gmf = self.gmf(user_idx, item_idx)    # (batch_size, embed_dim)
        x_mlp = self.mlp(user_idx, item_idx)    # (batch_size, hidden_dim)
        x = torch.cat((x_gmf, x_mlp), dim=1)    # (batch_size, embed_dim+hidden_dims[-1])
        x = self.nmf(x)                         # (batch_size, 1)
        return x.squeeze()                      # (batch_size, )

class GMF(nn.Module):
    def __init__(self, config):
        super().__init__()
       
        # hyper-parameters
        self.config = config
        self.emb_dim = config['emb_dim']
        self.nb_users = config['user_cnts']
        self.nb_items = config['item_cnts']

        # embedding layers
        self.emb_user = nn.Embedding(num_embeddings=self.nb_users, embedding_dim=self.emb_dim)
        self.emb_item = nn.Embedding(num_embeddings=self.nb_items, embedding_dim=self.emb_dim)
        
    def forward(self, user_idx, item_idx):
        user_emb = self.emb_user(user_idx)  # (batch_size, embed_dim)
        item_emb = self.emb_item(item_idx)  # (batch_size, embed_dim)
        product = user_emb * item_emb       # (batch_size, embed_dim)
        return product

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
       
        # hyper-parameters
        self.config = config
        self.emb_dim = config['emb_dim']
        self.nb_users = config['user_cnts']
        self.nb_items = config['item_cnts']
        self.hidden_dims = config['hidden_dims']

        # embedding layers
        self.emb_user = nn.Embedding(num_embeddings=self.nb_users, embedding_dim=self.emb_dim)
        self.emb_item = nn.Embedding(num_embeddings=self.nb_items, embedding_dim=self.emb_dim)
        
        # mlp layer
        layers = []
        in_dim = 2 * self.emb_dim
        for out_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, user_idx, item_idx):
        user_emb = self.emb_user(user_idx)          # (batch_size, embed_dim)
        item_emb = self.emb_item(item_idx)          # (batch_size, embed_dim)
        x = torch.cat((user_emb, item_emb), dim=1)  # (batch_size, 2*embed_dim)
        x = self.mlp(x)                             # (batch_size, hidden_dims[-1])
        return x
