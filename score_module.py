import torch
import torch.nn as nn
from torch.nn import functional as F

class ScoreMatrixPostProcessor(nn.Module):
    def __init__(self, nTarget, nHidden, dropoutProb):
        super().__init__()

        self.map = nn.Sequential(
                nn.Conv2d(nTarget, nHidden, 3, padding=2),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Conv2d(nHidden, nTarget, 3))

    def forward(self, S):

        S = S.permute(2, 3, 0, 1)
        S = self.map(S)
        S = S.permute(2, 3, 0, 1).contiguous()
        return S

class pairwise_score_module(nn.Module):
    def __init__(self,
                 inputSize,
                 outputSize,
                 dropoutProb = 0.0,
                 lengthScaling=False,
                 postConv=False,
                 hiddenSize=None,
                 moments=True, 
                 skip_score=False):
        super().__init__()

        self.skip_score = skip_score
        if hiddenSize is None:
            hiddenSize = outputSize * 4
            
        self.scoreMap = nn.Sequential(nn.Linear(inputSize*6, hiddenSize),
                                      nn.GELU(),
                                      nn.Dropout(dropoutProb),
                                      nn.Linear(hiddenSize, outputSize))
        
        self.scoreMapSkip = nn.Sequential(
                nn.Linear(inputSize*3, hiddenSize),
                nn.GELU(),
                nn.Dropout(dropoutProb),
                nn.Linear(hiddenSize, outputSize))

        self.lengthScaling = lengthScaling
        
        self.post = nn.Identity()
        if postConv:
            self.post = ScoreMatrixPostProcessor(outputSize, outputSize*3, dropoutProb)

    def compute_chunk(self, x, x_cum, x_sqr_cum,x_cube_cum, idxA, idxB):
        # A: end
        # B: begin
        curA = x[idxA]
        curB = x[idxB]

        lengthBA = (idxA - idxB) + 1
        lengthBA = lengthBA.view(-1, 1, 1)

        moment1 = (x_cum[idxA+1]- x_cum[idxB])/lengthBA
        moment2 = (x_sqr_cum[idxA+1]- x_sqr_cum[idxB])/lengthBA
        moment3 = (x_cube_cum[idxA+1]- x_cube_cum[idxB])/lengthBA

        curInput = torch.cat([curA, curB,  curA*curB, moment1, moment2, moment3], dim = -1)
        curScore = self.scoreMap(curInput)

        return curScore
    
    def compute_skip_score(self, x):
        curA = x[:-1]
        curB = x[1:]
        curInput = torch.cat([curA, curB,  curA*curB], dim=-1)
        curScore = self.scoreMapSkip(curInput)
        return curScore

    def forward(self, x, nBlock=4000):

        # input shape: [time_step, batch_size, embedding_dim]
        assert(len(x.shape)==3)
        x = x.transpose(0, 1)
        n_timestep = x.shape[0]
        indices = torch.tril_indices(n_timestep, n_timestep, device=x.device)
        n_total = indices.shape[1]
        S_all = []
        
        # padding means: pad the 3rd last dim (first dimension) by 1 on each side
        x_cum = torch.cumsum(F.pad(x, (0, 0, 0, 0, 1, 0)), dim=1) 
        x_sqr_cum = torch.cumsum(F.pad(x.pow(2), (0, 0, 0, 0, 1, 0)), dim=0)
        x_cube_cum = torch.cumsum(F.pad(x.pow(3), (0, 0, 0, 0, 1, 0)), dim=0)
        
        for lIdx in range(0, n_total, nBlock):
            if lIdx+nBlock< n_total:
                idxA = indices[0, lIdx:lIdx+nBlock]
                idxB = indices[1, lIdx:lIdx+nBlock]
            else:
                idxA = indices[0, lIdx:]
                idxB = indices[1, lIdx:]

            curScore = self.compute_chunk(x, x_cum, x_sqr_cum, x_cube_cum, idxA, idxB)
            S_all.append(curScore)
        
        s_val = torch.cat(S_all, dim=0)
        
        S_coo = torch.sparse_coo_tensor(indices, s_val, 
                                        (n_timestep, n_timestep, s_val.shape[-2], s_val.shape[-1]))

        S = S_coo.to_dense()

        S =  self.post(S)

        if self.lengthScaling:
            tmpIdx = torch.arange(nEntry, device = S.device)
            lenBA = (tmpIdx.unsqueeze(-1)- tmpIdx.unsqueeze(0)).abs().clamp(1)
            S = lenBA.unsqueeze(-1).unsqueeze(-1)*S
        
        S = S.flatten(-2, -1)
        if self.skip_score:
            S_skip = self.compute_skip_score(x).flatten(-2, -1)
        else:
            S_skip = torch.randn(S.shape[0]-1, S.shape[-1]).to(S.device) * 0
        
        return S, S_skip
    