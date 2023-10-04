import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from utils.utils import initialize_weights
import numpy as np


"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_tasks = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_tasks)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


"""
TOAD multi-task + concat mil network w/ attention-based pooling
args:
    gate: whether to use gating in attention network
    size_args: size config of attention network
    dropout: whether to use dropout in attention network
    n_classes: number of classes
"""

class ROAD_fc_mtl_concat(nn.Module):
    def __init__(self, gate = True, size_arg = "big", dropout = False, n_classes = 2, gene = True, cli = False):
        super(ROAD_fc_mtl_concat, self).__init__()
        self.gene = gene
        self.cli = cli
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        fc.extend([nn.Linear(size[1], size[1]), nn.ReLU()])
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = dropout, n_tasks = 1)
        else:
            attention_net = Attn_Net(L = size[1], D = size[2], dropout = dropout, n_tasks = 2)
        
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        if (gene == True) and (cli == True):
            self.classifier = nn.Linear(size[1] + 219 + 33, n_classes)# + 219
        elif gene == True:
            self.classifier = nn.Linear(size[1] + 1086, n_classes)
        elif cli == True:
            self.classifier = nn.Linear(size[1] + 33, n_classes)
        else:
            self.classifier = nn.Linear(size[1], n_classes)
#         self.site_classifier = nn.Linear(size[1], 2)

        initialize_weights(self)
                
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        else:
            self.attention_net = self.attention_net.to(device)


        self.classifier = self.classifier.to(device)
#         self.site_classifier = self.site_classifier.to(device)
        
    def forward(self, h, gene, cli, return_features=True, attention_only=False):
        A, h = self.attention_net(h)  
        A = torch.transpose(A, 1, 0) 
        if attention_only:
            return A[0]
        
        A_raw = A 
        A = F.softmax(A, dim=1) 
        M = torch.mm(A, h)
        if (self.gene == True) and (self.cli == True):
            M = torch.cat([M, gene.repeat(M.size(0), 1)], dim=1)
            M = torch.cat([M, cli.repeat(M.size(0), 1)], dim=1)
        if self.gene == True:
            M = torch.cat([M, gene.repeat(M.size(0), 1)], dim=1)
        if self.cli == True:
            M = torch.cat([M, cli.repeat(M.size(0), 1)], dim=1)
        try:
            logits  = self.classifier(M[0].unsqueeze(0))
        except:
            print(gene.shape)
            print(cli.shape)
            raise(NameError)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)


        results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        
        results_dict.update({'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat, 'A': A_raw})

        return results_dict

class Total_Loss(nn.Module):
    def __init__(self):
        super(Total_Loss, self).__init__()
        
    def forward(self, y_pred, event):
        partial_hazard = torch.exp(y_pred)
        log_cum_partial_hazard = torch.log(torch.cumsum(partial_hazard, dim=0))
        event_likelihood = (y_pred - log_cum_partial_hazard) * event
        neg_likelihood = -1.0 * torch.sum(event_likelihood)

        return neg_likelihood
    
class Regularization(object):
    def __init__(self, order, weight_decay):
        ''' The initialization of Regularization class

        :param order: (int) norm order number
        :param weight_decay: (float) weight decay rate
        '''
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def __call__(self, model):
        ''' Performs calculates regularization(self.order) loss for model.

        :param model: (torch.nn.Module object)
        :return reg_loss: (torch.Tensor) the regularization(self.order) loss
        '''
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss = reg_loss + torch.norm(w, p=self.order)
        reg_loss = self.weight_decay * reg_loss
        return reg_loss

class DeepSurv(nn.Module):
    ''' The module class performs building network according to config'''
    def __init__(self, config):
        super(DeepSurv, self).__init__()
        # parses parameters of network from configuration
        self.drop = config['drop']
        self.norm = config['norm']
        self.dims = config['dims']
        self.activation = config['activation']
        # builds network
        self.model = self._build_network()

    def _build_network(self):
        ''' Performs building networks according to parameters'''
        layers = []
        for i in range(len(self.dims)-1):
            if i and self.drop is not None: # adds dropout layer
                layers.append(nn.Dropout(self.drop))
            # adds linear layer
            layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
            if self.norm: # adds batchnormalize layer
                layers.append(nn.BatchNorm1d(self.dims[i+1]))
            # adds activation layer
            layers.append(eval('nn.{}()'.format(self.activation)))
        # builds sequential network
        return nn.Sequential(*layers)

    def forward(self, X):
        return self.model(X)

class NegativeLogLikelihood(nn.Module):
    def __init__(self, config=0):
        super(NegativeLogLikelihood, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.L2_reg = config
        self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    def forward(self, risk_pred, y, e, model):
        mask = torch.ones(y.shape[0], y.shape[0])
        mask[(y.T - y) > 0] = 0
        mask = mask.to(self.device)
        log_loss = torch.exp(risk_pred) * mask
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(mask, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / (torch.sum(e) + 1)
        l2_loss = self.reg(model)
        return neg_log_loss + l2_loss