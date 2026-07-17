from utlis import *
import math
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_net = nn.Sequential(
            nn.Linear(320, 160),
            nn.LeakyReLU(),
            nn.Linear(160, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 2),
        )

    def forward(self, x):
        logits = self.output_net(x)
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        print(self.pe[:x.size(0)].shape)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MLPwAttn(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(input_dim, embed_dim)
        # self.positional_encoding = PositionalEncoding(d_model=embed_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)
        self.linear_net = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim/2)),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(int(embed_dim/2), embed_dim)
        )
        self.output_net = nn.Sequential(
            nn.Linear(320, 160),
            nn.LeakyReLU(),
            nn.Linear(160, 40),
            nn.LeakyReLU(),
            nn.Linear(40, 2),
        )
        self.squeeze = nn.Linear(embed_dim, 1)
        self.flatten = nn.Flatten()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x) * math.sqrt(self.embed_dim)
        # x = self.positional_encoding(x)
        attn_out, attn_weight = self.multihead_attn(x, x, x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        linear_out = self.linear_net(x)
        x = x + self.dropout2(linear_out)
        x = self.norm2(x)

        x = self.squeeze(x)
        x = self.flatten(x)
        y = self.output_net(x)
        return y, attn_weight


class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.weight = Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # Mem: MxC
        self.bias = None
        self.shrink_thres = shrink_thres
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        att_weight = F.linear(input, self.weight)  # Fea x Mem^T: (TxC) x (CxM) = TxM
        att_weight = F.softmax(att_weight, dim=1)  # TxM
        # ReLU based shrinkage, hard shrinkage for positive value
        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)
        mem_trans = self.weight.permute(1, 0)  # Mem^T: C x M
        output = F.linear(att_weight, mem_trans)  # AttWeight x (Mem^T)^T: (TxM) x (MxC) = TxC
        return {'output': output, 'att': att_weight}  # output, att_weight

    def extra_repr(self):
        return 'mem_dim={}, fea_dim={}'.format(
            self.mem_dim, self.fea_dim is not None
        )


# NxCxL -> (NxL)xC -> addressing Mem, (NxL)xC -> NxCxL
class MemModule(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres):
        super(MemModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.memory = MemoryUnit(self.mem_dim, self.fea_dim, self.shrink_thres)

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 3:
            x = input.permute(0, 2, 1)
        elif l == 4:
            x = input.permute(0, 2, 3, 1)
        elif l == 5:
            x = input.permute(0, 2, 3, 4, 1)
        else:
            x = []
            print('wrong feature map size')
        
        x = x.contiguous()
        x = x.view(-1, s[1])
        y_and = self.memory(x)
        y = y_and['output']
        att = y_and['att']

        if l == 3:
            y = y.view(s[0], s[2], s[1])
            y = y.permute(0, 2, 1)
            att = att.view(s[0], s[2], self.mem_dim)
            att = att.permute(0, 2, 1)
        elif l == 4:
            y = y.view(s[0], s[2], s[3], s[1])
            y = y.permute(0, 3, 1, 2)
            att = att.view(s[0], s[2], s[3], self.mem_dim)
            att = att.permute(0, 3, 1, 2)
        elif l == 5:
            y = y.view(s[0], s[2], s[3], s[4], s[1])
            y = y.permute(0, 4, 1, 2, 3)
            att = att.view(s[0], s[2], s[3], s[4], self.mem_dim)
            att = att.permute(0, 4, 1, 2, 3)
        else:
            y = x
            att = att
            print('wrong feature map size')

        return {'output': y, 'att': att}
    

# NxCxHxW -> (HxW)xCxN -> truncated SVD -> (HxW)xCxN -> NxCxHxW 
class TSVDModule(nn.Module):
    def __init__(self, low_rank):
        super(TSVDModule, self).__init__()
        self.low_rank = low_rank

    def forward(self, input):
        s = input.data.shape
        l = len(s)

        if l == 4:
            x = input.permute(2, 3, 1, 0)
        else:
            x = []
            print('wrong feature map size')

        x = x.contiguous()
        x = x.view(-1, s[1], s[0])
        U, S, Vh = torch.svd_lowrank(x, q=self.low_rank)
        y = U @ torch.diag_embed(S) @ torch.transpose(Vh, 1, 2)

        if l == 4:
            y = y.view(s[2], s[3], s[1], s[0])
            y = y.permute(3, 2, 0, 1)
        else:
            y = x
            att = att
            print('wrong feature map size')

        return y


class MemAE_1d(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres, r):
        super().__init__()
        print('Model: MemAE_1d')
        self.chnum_in = chnum_in
        self.r = r
        feature_num_1 = 2
        feature_num_2 = 4
        feature_num_3 = 8
        feature_num_4 = 16
        self.encoder = nn.Sequential(
            nn.Conv1d(self.chnum_in, feature_num_1, 3, stride=2, padding=1),
            nn.BatchNorm1d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_1, feature_num_2, 3, stride=2, padding=1),
            nn.BatchNorm1d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_2, feature_num_3, 3, stride=2, padding=1),
            nn.BatchNorm1d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_3, feature_num_4, 3, stride=2, padding=1),
            nn.BatchNorm1d(feature_num_4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # self.memory_bank = MemModule(mem_dim=mem_dim, fea_dim=feature_num_4, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_num_4, feature_num_3, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_3, feature_num_2, 3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm1d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_2, feature_num_1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_1, self.chnum_in, 3, stride=2, padding=1, output_padding=0)
        )

    def forward(self, x):
        z = self.encoder(x)

        s = z.data.shape
        z_fw = z[:, :, 0:self.r]
        z_rem = z[:, :, self.r:]
        z_fw = z_fw.contiguous()
        z_fw = z_fw.view(s[0], -1)

        # res_mem = self.memory_bank(z)
        # z_hat = res_mem['output']
        # tt_weight = res_mem['att']
        output = self.decoder(z)
        return output, z_rem, z_fw, z


class MemAE_1d_10(nn.Module):
    def __init__(self, chnum_in, mem_dim, shrink_thres, r):
        super().__init__()
        print('Model: MemAE_1d_10')
        self.chnum_in = chnum_in
        self.r = r
        feature_num_1 = 2
        feature_num_2 = 4
        feature_num_3 = 8
        feature_num_4 = 16
        self.encoder = nn.Sequential(
            nn.Conv1d(self.chnum_in, feature_num_1, 11, stride=2, padding=5),
            nn.BatchNorm1d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_1, feature_num_2, 11, stride=2, padding=5),
            nn.BatchNorm1d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_2, feature_num_3, 11, stride=2, padding=5),
            nn.BatchNorm1d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(feature_num_3, feature_num_4, 11, stride=2, padding=5),
            nn.BatchNorm1d(feature_num_4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # self.memory_bank = MemModule(mem_dim=mem_dim, fea_dim=feature_num_4, shrink_thres=shrink_thres)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(feature_num_4, feature_num_3, 11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_3, feature_num_2, 11, stride=2, padding=5, output_padding=0),
            nn.BatchNorm1d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_2, feature_num_1, 11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(feature_num_1, self.chnum_in, 11, stride=2, padding=5, output_padding=0)
        )

    def forward(self, x):
        z = self.encoder(x)

        s = z.data.shape
        z1 = z[:, :, :self.r]
        z1 = z1.contiguous()
        z1 = z1.view(s[0], -1)

        # res_mem = self.memory_bank(z)
        # z_hat = res_mem['output']
        # tt_weight = res_mem['att']
        output = self.decoder(z)
        return output, s, z, z1


class MemAE_2d(nn.Module):
    def __init__(self, chnum_in, kernel_size, mem_dim, shrink_thres, low_rank, low_rank_processing):
        super(MemAE_2d, self).__init__()
        print('Model: MemAE_2d')
        self.chnum_in = chnum_in
        self.low_rank_processing_ = low_rank_processing
        feature_num_1 = 16
        feature_num_2 = 64
        feature_num_3 = 128
        # feature_num_4 = 512
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_1, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num_3, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(feature_num_3, feature_num_4, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            # nn.BatchNorm2d(feature_num_4),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        self.low_rank_processing = TSVDModule(low_rank=low_rank)
        self.memory_processing = MemModule(mem_dim=mem_dim, fea_dim=feature_num_3, shrink_thres=shrink_thres)
        if kernel_size == 3:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(feature_num_3, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(feature_num_2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_2, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(feature_num_1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_1, self.chnum_in, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1))
            )
        elif kernel_size == 5:
            self.decoder = nn.Sequential(
                # nn.ConvTranspose2d(feature_num_4, feature_num_3, (3, 3), stride=(2, 2), padding=(1, 1),
                #                    output_padding=(1, 1)),
                # nn.BatchNorm2d(feature_num_3),
                nn.ConvTranspose2d(feature_num_3, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(0, 0)),
                nn.BatchNorm2d(feature_num_2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_2, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(0, 0)),
                nn.BatchNorm2d(feature_num_1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_1, self.chnum_in, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1))
            )

    def forward(self, x):
        z = self.encoder(x)

        if self.low_rank_processing_:
            z_low_rank = self.low_rank_processing(z)
            # z1 = torch.mean(z, dim=0)
            # z2 = z1.unsqueeze(0).repeat(z.shape[0], 1, 1, 1)
            res_mem = self.memory_processing(z_low_rank)
        else:
            res_mem = self.memory_processing(z)
        
        z_hat = res_mem['output']

        # s = z_hat.data.shape
        # p = z_hat.permute(2, 3, 1, 0)
        # p = p.contiguous()
        # p = p.view(-1, s[1], s[0])

        att_weight = res_mem['att']
        output = self.decoder(z_hat)
        return output, att_weight

    
class LRAE_2d(nn.Module):
    def __init__(self, chnum_in, kernel_size):
        super(LRAE_2d, self).__init__()
        print('Model: LRAE_2d')
        self.chnum_in = chnum_in
        feature_num_1 = 16
        feature_num_2 = 64
        feature_num_3 = 128
        self.encoder = nn.Sequential(
            nn.Conv2d(self.chnum_in, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_1, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_num_2, feature_num_3, (kernel_size, kernel_size), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(feature_num_3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        if kernel_size == 3:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(feature_num_3, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(feature_num_2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_2, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(feature_num_1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_1, self.chnum_in, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1))
            )
        elif kernel_size == 5:
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(feature_num_3, feature_num_2, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(0, 0)),
                nn.BatchNorm2d(feature_num_2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_2, feature_num_1, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(0, 0)),
                nn.BatchNorm2d(feature_num_1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(feature_num_1, self.chnum_in, (kernel_size, kernel_size), stride=(2, 2), 
                                   padding=(1, 1), output_padding=(1, 1))
            )

    def forward(self, x):
        z = self.encoder(x)

        s = z.data.shape
        p = z.permute(2, 3, 1, 0)
        p = p.contiguous()
        p = p.view(-1, s[1], s[0])

        output = self.decoder(z)
        return output, p

    

