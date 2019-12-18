from __future__ import absolute_import 
import torch 
import torch.nn as nn 
from torch.autograd import Variable 

class TripletLoss(nn.Module):
    def __init__(self, margin=1.2):
        super(TripletLoss, self).__init__()
        self.margin = margin 
        self.ranking_loss = nn.MarginRankingLoss(margin=self.margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        dist = self.Euclidean_Distance(inputs, inputs)
        mask = targets.expand(n,n).eq(targets.expand(n,n).t())
        dist_ap, dist_an = [], [] 

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i]==0].min())
        
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)

        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss 


    def Euclidean_Distance(self, x1, x2):
        n, m = x1.size(0), x2.size(0)
        squared_x1 = torch.pow(x1, 2).sum(dim=1, keepdim=True).expand(n, m)
        squared_x2 = torch.pow(x2, 2).sum(dim=1, keepdim=True).expand(m, n)
        dist = squared_x1 + squared_x2.t() 
        dist.addmm_(1, -2, x1, x2.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist 
