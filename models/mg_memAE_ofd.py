import math
from models.basic_modules import *
from models.convnext import Block
from models.series import series_decomp
# relu based hard shrinkage function, only works for positive values
def hard_shrink_relu(input, lambd=0., epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class Memory(nn.Module):
    def __init__(self, num_slots, slot_dim, shrink_thres=0.0025):
        super(Memory, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        self.memMatrix = nn.Parameter(torch.empty(num_slots, slot_dim))  # M,C
        self.shrink_thres = shrink_thres

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.memMatrix.size(1))
        self.memMatrix.data.uniform_(-stdv, stdv)

    def forward(self, x):

 
        att_weight = F.linear(input=x, weight=self.memMatrix)  
        att_weight = F.softmax(att_weight, dim=1)

        # if use hard shrinkage
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            # normalize
            att_weight = F.normalize(att_weight, p=1, dim=1) 

        # out slot
        out = F.linear(att_weight, self.memMatrix.permute(1, 0))

        return dict(out=out, att_weight=att_weight)

class MG_MemAE_OFD(nn.Module):
    def __init__(self, num_in_ch, seq_len, features_root,
                 num_slots, shrink_thres,drop_rate,
                 mem_usage, skip_ops):
        super(MG_MemAE_OFD, self).__init__()
        self.num_in_ch = num_in_ch
        self.seq_len = seq_len
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.mem_usage = mem_usage
        self.num_mem = sum(mem_usage)
        self.skip_ops = skip_ops
        self.serise=series_decomp(2)
        self.conv1= nn.Conv1d(in_channels=32*32,out_channels=32*32,kernel_size=3,padding=1)

        self.convnextblock1=Block(features_root,drop_rate)
        self.convnextblock2=Block(features_root * 2,drop_rate)
        self.convnextblock3=Block(features_root * 4,drop_rate)

        self.in_conv = inconv(num_in_ch * seq_len, features_root)
        self.down_1 = down(features_root, features_root * 2)
        self.down_2 = down(features_root * 2, features_root * 4)
        self.down_3 = down(features_root * 4, features_root * 8)

        # memory modules
        self.mem1 = Memory(num_slots=self.num_slots, slot_dim=features_root * 2 * 16 * 16,
                           shrink_thres=self.shrink_thres) if self.mem_usage[1] else None
        self.mem2 = Memory(num_slots=self.num_slots, slot_dim=features_root * 4 * 8 * 8,
                           shrink_thres=self.shrink_thres) if self.mem_usage[2] else None
        self.mem3 = Memory(num_slots=self.num_slots, slot_dim=features_root * 8 * 4 * 4,
                           shrink_thres=self.shrink_thres) if self.mem_usage[3] else None

        self.up_3 = up(features_root * 8, features_root * 4, op=self.skip_ops[-1])
        self.up_2 = up(features_root * 4, features_root * 2, op=self.skip_ops[-2])
        self.up_1 = up(features_root * 2, features_root, op=self.skip_ops[-3])
        self.out_conv = outconv(features_root, num_in_ch * seq_len)

    def forward(self, x):
        """
        :param x: size [bs,C*seq_len,H,W]
        :return:
        """
        x, mean = self.serise(x)


        x0=self.in_conv(x)#4 32 32 32
        x1=self.down_1(self.convnextblock1(x0))#4 64 16 16
        x2=self.down_2(self.convnextblock2(x1))#4 128 8 8
        x3=self.down_3(self.convnextblock3(x2))#4 128 4 4
        if self.mem_usage[3]:#true
            # flatten [bs,C,H,W] --> [bs,C*H*W]
            bs, C, H, W = x3.shape
            x3 = x3.view(bs, -1)
            mem3_out = self.mem3(x3)
            x3 = mem3_out["out"]
            # attention weight size [bs,N], N is num_slots
            att_weight3 = mem3_out["att_weight"]
            # unflatten
            x3 = x3.view(bs, C, H, W)


        bs, C, H, W = x2.shape
        x2 = x2.view(bs, -1)
        mem2_out = self.mem2(x2)
        x2 = mem2_out["out"]
        # attention weight size [bs,N], N is num_slots
        att_weight2 = mem2_out["att_weight"]
        # unflatten
        x2 = x2.view(bs, C, H, W)
        recon=self.convnextblock3(self.up_3(x3, x2 if self.skip_ops[-1] != "none" else None))


        bs, C, H, W = x1.shape#4 64 16 16
        x1 = x1.view(bs, -1)#4,8192
        mem1_out = self.mem1(x1)
        x1 = mem1_out["out"]
        att_weight1 = mem1_out["att_weight"]
        x1 = x1.view(bs, C, H, W)
        recon=self.convnextblock2(self.up_2(recon, x1 if self.skip_ops[-2] != "none" else None))

        recon=self.convnextblock1(self.up_1(recon, x0 ))
        recon = self.out_conv(recon)
        bm,cm,wm,dm=mean.shape
        mean=self.conv1(mean.reshape(bm,cm,-1).permute(0,2,1)).permute(0,2,1).reshape(bm,cm,wm,dm)
        recon=recon+mean


        if self.num_mem == 3:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2, att_weight1=att_weight1)
        elif self.num_mem == 2:
            outs = dict(recon=recon, att_weight3=att_weight3, att_weight2=att_weight2,
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        elif self.num_mem == 1:
            outs = dict(recon=recon, att_weight3=att_weight3,
                        att_weight2=torch.zeros_like(att_weight3),
                        att_weight1=torch.zeros_like(att_weight3))  # dummy attention weights
        return outs




if __name__ == '__main__':
    model = MG_MemAE_OFD(num_in_ch=2, seq_len=1, features_root=32, num_slots=2000, shrink_thres=1 / 2000,drop_rate=0,
                        mem_usage=[False, True, True, True], skip_ops=["none", "concat", "concat"])
    dummy_x = torch.rand(4, 2, 32, 32)
    dummy_out = model(dummy_x)
    recon=dummy_out["recon"]
    print(recon.size())
    print(dummy_out)
    print(-1)
