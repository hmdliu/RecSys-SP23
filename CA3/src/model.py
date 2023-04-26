
import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
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

    def forward(self, u, i, js):
        # compute x_ui
        emb_u = self.emb_user(u)                            # (batch_size, embed_dim)
        emb_i = self.emb_item(i)                            # (batch_size, embed_dim)
        x_ui = torch.mul(emb_u, emb_i).sum(dim=1)           # (batch_size, )
        # compute x_uj for all j
        if len(js.shape) > 1:
            x_uj = 0
            for j in range(js.shape[-1]):
                emb_j = self.emb_item(js[:, j])             # (batch_size, embed_dim)
                x_uj += torch.mul(emb_u, emb_j).sum(dim=1)  # (batch_size, )
            x_uj /= js.shape[-1]                            # (batch_size, )
        else:
            emb_j = self.emb_item(js)                       # (batch_size, embed_dim)
            x_uj = torch.mul(emb_u, emb_j).sum(dim=1)       # (batch_size, )
        # compute logits
        x_uij = x_ui - x_uj                                 # (batch_size, )
        return torch.sigmoid(x_uij)
    
    def predict(self, u, i):
        # compute x_ui
        emb_u = self.emb_user(u)                            # (batch_size, embed_dim)
        emb_i = self.emb_item(i)                            # (batch_size, embed_dim)
        x_ui = torch.mul(emb_u, emb_i).sum(dim=1)           # (batch_size, )
        return torch.sigmoid(x_ui)

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
