import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from . import embutils

_dir = os.path.dirname(os.path.abspath(__file__))

class Args:
    def __init__(self, model):
        if model == "s24":
            self.lr = 0.001
            self.epochs = 256
            self.batch_size=64
            self.dropout=0.5
            self.max_norm=3.0
            self.embed_dim=128
            self.class_num = 3
            self.cuda = False
            self.kernel_sizes = [2,3,4,5]
            self.snapshot = os.path.join(_dir, 's24_sentiment/final_model.pt')
            self.kernel_num = 100
            self.device = -1
            self.static=True

# model
class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args
        self.embs = embutils.WordEmbeddings()
        self.embs.load_from_file(os.path.join(_dir,
                                              "s24_sentiment/s24_surface_vecs.bin"))

        self.steps = 0
        
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def embed(self, texts):
        veclists = []
        for text in texts:
            veclists.append([])
            for token in text:
                emb = self.embs.get_embedding(token)
                veclists[-1].append(emb[1])
        def pad(veclist, to, padding):
            while len(veclist) < to:
                veclist.append(padding)
        maxlen = max(max(map(len, texts)), max(self.args.kernel_sizes))
        padding = [0.0] * self.args.embed_dim
        for veclist in veclists:
            pad(veclist, maxlen, padding)
        return torch.Tensor(veclists)
        
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = list(x)
        x = self.embed(x)  # (N, W, D)
        
        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        #logit = torch.squeeze(self.fc1(x))  # (N, 1)
        logit = self.fc1(x)  # (N, C)
        return logit

    def list(self, inputs):
        preds = ["neg", "neut", "pos"]
        retval = []
        for i, prediction in enumerate(self(inputs)):
            retval.append(preds[torch.argmax(prediction)])
        return retval

    def txt(self, inputs):
        return '\n'.join(self.list(inputs))

s24_args = Args("s24")
s24 = CNN_Text(s24_args)
s24.load_state_dict(torch.load(s24_args.snapshot))
s24.eval()
