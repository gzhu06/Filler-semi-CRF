import torch.nn as nn
from score_module import *
import CRF
from s4.s4 import S4

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
class Block1(nn.Module):
    def __init__(self, cin, cout, kernel_size=9, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin,
                      cout,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=padding),
            nn.BatchNorm1d(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv = self.block(x)
        x = x + x_conv
        outputs = self.relu(x)
        return outputs
    
class Block2(nn.Module):
    def __init__(self, cin, cout, 
                 kernel_size=9, 
                 stride_size=2,
                 pad_size=4):
        super().__init__()
        self.mainblock = nn.Sequential(
            nn.Conv1d(cin, cout,
                      kernel_size=kernel_size,
                      stride=stride_size,
                      padding=pad_size),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
            nn.Conv1d(cout,
                      cout,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=pad_size),
            nn.BatchNorm1d(cout))
        
        self.sideblock = nn.Sequential(
            nn.Conv1d(cin,
                      cout,
                      kernel_size=1,
                      stride=stride_size,
                      padding=0),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_conv1 = self.mainblock(x)
        x_conv2 = self.sideblock(x)
        x_conv = x_conv1 + x_conv2
        outputs = self.relu(x_conv)
        return outputs
    
class S4_layer(nn.Module):
    def __init__(self, dim, state_dim, bidirectional, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.S4 = S4(d_model=dim, d_state=state_dim, 
                     bidirectional=bidirectional, 
                     dropout=dropout)

    def forward(self, x):
        x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.S4(x)[0]
        
        return x

    
class SEGBASELINE(nn.Module):
    def __init__(self, inputdim, numclass=6, labelres=20, 
                 backbone_type='crnn', nb_layers=4, **kwargs):
        super().__init__()
        
        self.inputdim = inputdim
        self.backbone_type = backbone_type
        
        if backbone_type == 's4':
            d_layers = []
            hidden_size = 64
            for _ in range(nb_layers):
                d_layers.append(S4_layer(dim=hidden_size, state_dim=64,
                                         bidirectional=True))

                
            self.s4_layer = nn.ModuleList(d_layers)
            self.outputlayer = nn.Linear(hidden_size, numclass)
        else:
            hidden_size = 32
            self.features = nn.Sequential(Block2(32, 48, stride_size=1),
                                          Block2(48, 48, stride_size=1),
                                          Block2(48, 64, stride_size=1),
                                          Block2(64, 64, stride_size=1))
            self.rnn = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
            self.features.apply(init_weights)
            self.outputlayer = nn.Linear(128, numclass)
        
        if labelres == 10:
            self.conv1 = nn.Conv1d(inputdim, hidden_size, kernel_size=3, stride=1, padding=1)
            self.apool = nn.AvgPool1d(10, stride=9, padding=0)
        elif labelres == 20:
            self.conv1 = nn.Conv1d(inputdim, hidden_size, kernel_size=3, stride=1, padding=3)
            self.apool = nn.AvgPool1d(10, stride=5, padding=2)
        
        # initialization
        self.conv1.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        
        x = self.conv1(x)
        
        if self.backbone_type == 's4':
            for layer in self.s4_layer:
                skip = x
                x = layer(x)
                x = x + skip
            
        else:
            x = self.features(x).transpose(1, 2)
            x, _ = self.rnn(x)
            x = x.transpose(1, 2)
        
        x = self.apool(x).transpose(1, 2)
        x = self.outputlayer(x)

        return x

class SEGCRF(nn.Module):
    def __init__(self, inputdim, numclass=6, nb_layers=4,
                 skip_score=False, backbone_type='crnn',
                 **kwargs):
        super().__init__()
        
        # parameters
        self.inputdim = inputdim
        self.numclass = numclass
        self.backbone_type = backbone_type
        
        # choose backbone
        if backbone_type == 'crnn':
            hidden_size = 32
            features = nn.ModuleList()
            self.features = nn.Sequential(Block2(32, 48, stride_size=1),
                                          Block2(48, 48, stride_size=1),
                                          Block2(48, 64, stride_size=1),
                                          Block2(64, 64, stride_size=1))
            self.rnn = nn.LSTM(64, 64, bidirectional=True, batch_first=True)
            self.pair_score = pairwise_score_module(128, numclass, skip_score=skip_score)
            self.features.apply(init_weights)
        elif backbone_type == 's4':
            d_layers = []
            hidden_size = 64
            for _ in range(nb_layers):
                d_layers.append(S4_layer(dim=hidden_size, state_dim=64,
                                         bidirectional=True))
            self.s4_layer = nn.ModuleList(d_layers)
            self.pair_score = pairwise_score_module(64, numclass, skip_score=skip_score)
        
        # n=20
        self.conv1 = nn.Conv1d(inputdim, hidden_size, kernel_size=3, stride=1, padding=3)
        self.apool = nn.AvgPool1d(10, stride=5, padding=2)

        # initialization
        self.conv1.apply(init_weights)

    def forward(self, x, y=None):
        
        x = self.conv1(x)
        
        if self.backbone_type == 's4':
            for layer in self.s4_layer:
                skip = x
                x = layer(x)
                x = x + skip
        
        elif self.backbone_type == 'crnn':
            x = self.features(x).transpose(1, 2)
            x, _ = self.rnn(x)
            x = x.transpose(1, 2)

        
        x = self.apool(x).transpose(1, 2)
    
        # compute pair score
        x, x_skip = self.pair_score(x)

        # compute crf logp
        crf = CRF.NeuralSemiCRFInterval(x, x_skip)
        if y is not None:
            y_flatten = sum(y, [])
            assert(len(y_flatten) == len(y)* self.numclass)
            pathScore  = crf.evalPath(y_flatten) 
            logZ = crf.computeLogZ()
            logProb = pathScore - logZ 
            logProb = logProb.view(len(y), -1)
        else:
            logProb = 0
        return logProb, crf
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    import os 
    
    fillerModel = SEGBASELINE(inputdim=64, numclass=6, backbone_type='s4')
    # fillerModel = SEGCRF(inputdim=64, numclass=6, backbone_type='s4')
    print(count_parameters(fillerModel))
    