

import torch
import torch.nn as nn



class Temporal(nn.Module):
    def __init__(self,dim,channel):
        super(Temporal, self).__init__()

        self.lin = nn.Linear(channel,channel*4,bias=True)
        self.sequence= nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=0, dilation=channel),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.InstanceNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, kernel_size=2, padding=0, dilation=channel),
            nn.Conv1d(dim, dim, 3, padding=1),
            nn.InstanceNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(channel, channel * 4, bias=True),
            nn.InstanceNorm1d(dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )


    def forward(self,x1,x2,x3,x4):
        x = torch.cat((x1,x2,x3,x4),dim=1)

        x0 = x.permute(0, 2, 1)
        x0 = self.sequence(x0)

        x0 = x0.permute(0, 2, 1)
        x = x0 + x
        x1 = x[:, 0:65, :]
        x2 = x[:, 65:130, :]
        x3 = x[:, 130:195, :]
        x4 = x[:, 195:260, :]
        return x1, x2, x3, x4

