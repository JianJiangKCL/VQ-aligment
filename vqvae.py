import torch
from torch import nn
from torch.nn import functional as F

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        # BxN_e
        embed = torch.randn(dim, n_embed)
        # self.my_buffer is from self.register_buffer(name, tensor)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        # BxHxWxC input
        # B*H*WxC
        flatten = input.reshape(-1, self.dim)
        # @ is matmul; dist is (x-y)^2 = x^2 - 2xy + y^2
        # B*H*WxN_e
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed  # B*H*WxC matmul* CxN_e => B*H*WxN_e
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        # values, indices B*H*W
        _, embed_ind = (-dist).max(1)
        # todo ; what's the usage of emb_onehot
        # B*H*WxN_e
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        # BxHxW
        embed_ind = embed_ind.view(*input.shape[:-1])
        # BxHxWxC ; todo where is the statement of self.embed_code
        quantize = self.embed_code(embed_ind)

        if self.training:

            # Note treat B*H*W as positions; C is a set of values in a certain position
            # B*H*WxN_e => N_e; it means how many times an embedding is chosen during this mini-batch
            embed_onehot_sum = embed_onehot.sum(0)
            # Note an embedding's value is learnt from its members' vector values (the avg of members' value)
            #   CxB*H*W matmul* B*H*WxN_e => CxN_e ;
            #   it means the summation of corresponding dim from selected members for an embedding;
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ####
            # cluster_size starts from 0; size is n_embed; (N_e)
            # an element inside means the size of (members) this cluster
            # Note for add_   out = self.data + alpha*other
            # decay is like learning rate for VQ.
            # as a whole,  cluster_size * decay + added_members * (1-decay) =>  cluster_size *0.99 + added_members +0.01
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            #####
            # #   an embedding is learnt from its old value and its members' value;

            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind

    def add_zero_emb(self, device):
        zero_embed = torch.zeros([self.dim,1]).to(device)
        self.embed = torch.cat([self.embed, zero_embed], dim=1)
        self.n_embed = self.n_embed + 1
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

# single level VQVAE
class VQVAE(nn.Module):
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512, # original is 512 =2^9, however 256 = 2^8, that can highly reduced
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4) # sp/4; p->p/4
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)    # sp/2; p/4->p/8
        # quantize_conv is used to change the channel size to the dimension of embedding vector
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(
            embed_dim, channel, channel, n_res_block, n_res_channel, stride=2
        ) # sp*2; p/8->p/4
        # self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        # self.quantize_b = Quantize(embed_dim, n_embed)
        # self.upsample_t = nn.ConvTranspose2d(
        #     embed_dim, embed_dim, 4, stride=2, padding=1
        # ) # sp *2; p/8->p/4
        self.dec = Decoder(
            channel,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        ) # sp*4; p/4 ->p

    def forward(self, input):
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_b, diff_b, id_b

    def encode_eva(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
        quant_t, diff_t, id_t = self.quantize_t(quant_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
        quant_b, diff_b, id_b = self.quantize_b(quant_b)
        quant_b = quant_b.permute(0, 3, 1, 2)
        diff_b = diff_b.unsqueeze(0)

        return quant_t, quant_b, diff_t , diff_b, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec

    def decode_code(self, code_t, code_b):
        # retrieve corresponding embeddings using HxW indices
        # BxCxHxW
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec