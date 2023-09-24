import torch
from torch import nn
from models.mg_memAE_ofd import MG_MemAE_OFD
from models.mae_fly import mae_vitnew


class OFST(nn.Module):

    def __init__(self, num_hist, num_pred, config, features_root, num_slots, shrink_thres, skip_ops, mem_usage,
                 finetune=False, drop_rate=0.):
        super(OFST, self).__init__()

        self.num_hist = num_hist
        self.num_pred = num_pred
        self.features_root = features_root
        self.num_slots = num_slots
        self.shrink_thres = shrink_thres
        self.skip_ops = skip_ops
        self.mem_usage = mem_usage
        self.finetune = finetune
        self.drop_rate=drop_rate

        self.x_ch = 3  # num of RGB channels
        self.y_ch = 2  # num of optical flow channels

        self.memAE = MG_MemAE_OFD(num_in_ch=self.y_ch, seq_len=1, features_root=self.features_root,
                                 num_slots=self.num_slots, shrink_thres=self.shrink_thres,
                                 mem_usage=self.mem_usage,
                                 skip_ops=self.skip_ops,
                                 drop_rate=self.drop_rate)

        self.mae_new=mae_vitnew()

        self.mse_loss = nn.MSELoss()

    def forward(self, sample_frame, sample_of, mode="train"):
        """
        :param sample_frame: 5 frames in a video clip
        :param sample_of: 4 corresponding flows
        :return:
        """
        att_weight3_cache, att_weight2_cache, att_weight1_cache = [], [], []
        # att_weightvit_cache=[]
        of_recon = torch.zeros_like(sample_of)

        # reconstruct flows
        for j in range(self.num_hist):
            memAE_out = self.memAE(sample_of[:, 2 * j:2 * (j + 1), :, :])
            of_recon[:, 2 * j:2 * (j + 1), :, :] = memAE_out["recon"]
            att_weight3_cache.append(memAE_out["att_weight3"])
            att_weight2_cache.append(memAE_out["att_weight2"])
            att_weight1_cache.append(memAE_out["att_weight1"])



        att_weight3 = torch.cat(att_weight3_cache, dim=0)
        att_weight2 = torch.cat(att_weight2_cache, dim=0)
        att_weight1 = torch.cat(att_weight1_cache, dim=0)

        if self.finetune:
            loss_recon = self.mse_loss(of_recon, sample_of)
            loss_sparsity = torch.mean(
                torch.sum(-att_weight3 * torch.log(att_weight3 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-att_weight2 * torch.log(att_weight2 + 1e-12), dim=1)
            ) + torch.mean(
                torch.sum(-att_weight1 * torch.log(att_weight1 + 1e-12), dim=1)
            )

        frame_in = sample_frame[:, :-self.x_ch * self.num_pred, :, :]
        frame_target = sample_frame[:, -self.x_ch * self.num_pred:, :, :]

        input_dict = dict(appearance=frame_in, motion=of_recon)
        frame_pred = self.mae_new(input_dict)


        out = dict(frame_pred=frame_pred, frame_target=frame_target,
                   of_recon=of_recon, of_target=sample_of)

        if self.finetune:
            MG_MemAE_OFD_dict = dict(loss_recon=loss_recon, loss_sparsity=loss_sparsity)
            out.update(MG_MemAE_OFD_dict)

        return out
