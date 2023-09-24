import torch
import torch.nn as nn


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series

        if self.kernel_size % 2 == 0:
            front = x[:, 0:1, :,:].repeat(1, (self.kernel_size) // 2, 1, 1)
            x = torch.cat([front, x], dim=1)
        else:
            front = x[:, 0:1, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
            end = x[:, -1:, :, :].repeat(1, (self.kernel_size - 1) // 2, 1, 1)
            x = torch.cat([front, x, end], dim=1)

        b, c, w, d = x.shape
        x = x.permute(0, 2, 3, 1).view(b,-1,c)
        x = self.avg(x)
        x = (x.view(b,w,d,-1)).permute(0, 3, 1, 2)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


if __name__ == "__main__":
    x = torch.rand(128,8,32,32)
    model=series_decomp(2)
    x_1,mean=model(x)
    print(x_1.shape)

