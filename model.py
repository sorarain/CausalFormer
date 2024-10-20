import math
import numpy as np
import torch
from strategy.svd import SVDAssignmentStrategy


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

class MultiHeadLinear(torch.nn.Module):
    def __init__(self,
                 num_heads:int,
                 input_size:int,
                 hidden_size:int):
        super(MultiHeadLinear, self).__init__()
        assert input_size % num_heads == 0
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = torch.nn.ModuleList([torch.nn.Linear(input_size // num_heads, hidden_size // num_heads) for _ in range(num_heads)])
    
    def forward(self, seqs):
        seqs_split = seqs.split(self.input_size // self.num_heads,dim=2)
        output = torch.zeros([0],device=seqs.device)
        for seq, layer in zip(seqs_split, self.linear):
            seq = (layer(seq))
            output = torch.cat([output, seq],dim=-1)
        return output


class NormalFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate, use_heads=False, num_heads=None):

        super(NormalFeedForward, self).__init__()

        if use_heads:
            self.conv1 = MultiHeadLinear(num_heads, hidden_units, hidden_units)
        else:
            self.conv1 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        if use_heads:
            self.conv2 = MultiHeadLinear(num_heads, hidden_units, hidden_units)
        else:
            self.conv2 = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs)))))
        # outputs = outputs.transpose(-1, -2)  # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# current: svd version
class ItemCode(torch.nn.Module):
    def __init__(self, pq_m, embedding_size, num_items, sequence_length, device):
        super(ItemCode, self).__init__()
        self.device = device
        self.pq_m = pq_m  # 8
        self.sub_embedding_size = embedding_size // self.pq_m  # 48 / 8
        self.item_code_bytes = embedding_size // self.sub_embedding_size  # 8
        self.vals_per_dim = 256
        self.base_type = torch.uint8
        self.item_codes = torch.zeros(
            size=(num_items + 1, self.item_code_bytes), dtype=self.base_type, device=self.device
        )  # trainable?
        self.centroids = torch.nn.Parameter(
            torch.randn(self.item_code_bytes, 256, self.sub_embedding_size, device=self.device)  # (8, 256, 6)
        )
        self.item_codes_strategy = SVDAssignmentStrategy(self.item_code_bytes, num_items, self.device)
        self.sequence_length = sequence_length
        self.num_items = num_items

    def assign_codes(self, train_users):
        code = self.item_codes_strategy.assign(train_users)
        self.item_codes = code

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)
        batch_size, sequence_length = input_ids.shape
        input_codes = self.item_codes[input_ids].detach().int()  # (256, 200, 8)
        code_byte_indices = torch.arange(self.item_code_bytes, device=self.device).unsqueeze(0).unsqueeze(0)
        code_byte_indices = code_byte_indices.repeat(batch_size, sequence_length, 1)
        n_sub_embeddings = batch_size * sequence_length * self.item_code_bytes
        code_byte_indices_reshaped = code_byte_indices.reshape(n_sub_embeddings)
        input_codes_reshaped = input_codes.reshape(n_sub_embeddings)
        indices = torch.stack([code_byte_indices_reshaped, input_codes_reshaped], dim=-1)
        input_sub_embeddings_reshaped = self.centroids[indices[:, 0], indices[:, 1]]
        result = input_sub_embeddings_reshaped.reshape(
            batch_size, sequence_length, self.item_code_bytes * self.sub_embedding_size
        )
        return result


# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py


class SASRec(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.pq_m = 8
        self.item_code = ItemCode(self.pq_m, args.hidden_units, item_num, args.maxlen, args.device)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def log2feats(
        self, log_seqs
    ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
        seqs = self.item_code(torch.LongTensor(log_seqs).to(self.dev))  # (256, 200) -> (256, 200, 48)
        # seqs *= self.item_emb.embedding_dim**0.5 # scaling?
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats

    # def log2feats(
    #     self, log_seqs
    # ):  # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
    #     seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
    #     seqs *= self.item_emb.embedding_dim**0.5
    #     poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
    #     # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
    #     poss *= log_seqs != 0
    #     seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
    #     seqs = self.emb_dropout(seqs)

    #     tl = seqs.shape[1]  # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for i in range(len(self.attention_layers)):
    #         seqs = torch.transpose(seqs, 0, 1)
    #         Q = self.attention_layernorms[i](seqs)
    #         mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
    #         # need_weights=False) this arg do not work?
    #         seqs = Q + mha_outputs
    #         seqs = torch.transpose(seqs, 0, 1)

    #         seqs = self.forward_layernorms[i](seqs)
    #         seqs = self.forward_layers[i](seqs)

    #     log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

    #     return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):  # for training
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        log_feats = self.log2feats(log_seqs)  # user_ids hasn't been used yet

        final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits  # preds # (U, I)

    def get_seq_embedding(self, input_ids):
        pass



# class MultiHeadSelfAttention(torch.nn.Module):
#     def __init__(self, embed_dim, num_heads, key_size, value_size, bias=False, use_heads=False, num_linear_heas=None):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.q_head_dim = key_size // num_heads
#         self.k_head_dim = key_size // num_heads
#         self.v_head_dim = value_size // num_heads

#         if not use_heads:
#             self.W_q = torch.nn.Linear(embed_dim, key_size, bias=bias)
#             self.W_k = torch.nn.Linear(embed_dim, key_size, bias=bias)
#             self.W_v = torch.nn.Linear(embed_dim, value_size, bias=bias)
#             self.out_proj = torch.nn.Linear(value_size, embed_dim, bias=bias)
#         else:
#             self.W_q = MultiHeadLinear(num_linear_heas, embed_dim, key_size)
#             self.W_k = MultiHeadLinear(num_linear_heas, embed_dim, key_size)
#             self.W_v = MultiHeadLinear(num_linear_heas, embed_dim, value_size)
#             self.out_proj = MultiHeadLinear(num_linear_heas, value_size, embed_dim)        

#         # self.q_proj = torch.nn.Linear(key_size, key_size, bias=bias)
#         # self.k_proj = torch.nn.Linear(key_size, key_size, bias=bias)
#         # self.v_proj = torch.nn.Linear(value_size, value_size, bias=bias)
        

#     def forward(self, x, attn_mask):
#         """
#         Args:
#             X: shape: (N, L, embed_dim), input sequence, 
#             是经过input embedding后的输入序列，L个embed_dim维度的嵌入向量

#         Returns:
#             output: (N, L, embed_dim)
#         """
#         query = self.W_q(x)  # (N, L, key_size)
#         key = self.W_k(x)  # (N, L, key_size)
#         value = self.W_v(x)  # (N, L, value_size)
#         # q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)
#         q, k, v = query, key, value
#         N, L, value_size = v.size()

#         q = q.reshape(N, L, self.num_heads, self.q_head_dim).transpose(1, 2)
#         k = k.reshape(N, L, self.num_heads, self.k_head_dim).transpose(1, 2)
#         v = v.reshape(N, L, self.num_heads, self.v_head_dim).transpose(1, 2)

#         att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(k.size(-1))
#         att = torch.nn.Softmax(dim=-1)(att)
#         # att = att * attn_mask
#         output = torch.matmul(att, v)
#         output = output.transpose(1, 2).reshape(N, L, value_size)
#         output = self.out_proj(output)

#         return output, att

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, heads, d_model, dropout=0.1, use_heads=False, alpha=0.0, num_linear_heas=None):
        super().__init__()
        self.d_model = d_model  # 模型的维度
        self.d_k = d_model // heads  # 每个头的维度
        self.h = heads  # 头的数量
        self.alpha = alpha

        # 以下三个是线性层，用于处理Q（Query），K（Key），V（Value）
        if use_heads:
            self.q_linear = MultiHeadLinear(num_linear_heas, d_model, d_model)
            self.v_linear = MultiHeadLinear(num_linear_heas, d_model, d_model)
            self.k_linear = MultiHeadLinear(num_linear_heas, d_model, d_model)
            self.out = MultiHeadLinear(num_linear_heas, d_model, d_model) 
        else:
            self.q_linear = torch.nn.Linear(d_model, d_model)
            self.v_linear = torch.nn.Linear(d_model, d_model)
            self.k_linear = torch.nn.Linear(d_model, d_model)
            self.out = torch.nn.Linear(d_model, d_model)  # 输出层

        self.dropout = torch.nn.Dropout(dropout)  # Dropout层
        

    def attention(self, q, k, v, d_k, attn_mask=None, causal_mask=None, dropout=None):
        # torch.matmul是矩阵乘法，用于计算query和key的相似度
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).repeat(scores.size(0),scores.size(1),1,1)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            else:
                assert 1 == 0
            scores = scores.masked_fill(attn_mask == 1, -1e9)  # 使用mask将不需要关注的位置设置为一个非常小的数
        if causal_mask is not None:
            # mask = mask.unsqueeze(1)  # 在第一个维度增加维度
            if causal_mask.dim() == 2:
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(scores.size(0),scores.size(1),1,1)
            elif causal_mask.dim() == 3:
                causal_mask = causal_mask.unsqueeze(1)
            else:
                assert 1 == 0
            scores *= (1 + self.alpha * causal_mask)

        # 对最后一个维度进行softmax运算，得到权重
        scores = torch.nn.functional.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)  # 应用dropout

        output = torch.matmul(scores, v)  # 将权重应用到value上
        return output

    def forward(self, q, k, v, attn_mask=None, causal_mask=None):
        bs = q.size(0)  # 获取batch_size

        # 将Q, K, V通过线性层处理，然后分割成多个头
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # 转置来获取维度为bs * h * sl * d_model的张量
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # 调用attention函数计算输出
        scores = self.attention(q, k, v, self.d_k, attn_mask, causal_mask, self.dropout)

        # 重新调整张量的形状，并通过最后一个线性层
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)  # 最终输出
        return output, scores

class EncoderLayer(torch.nn.Module):
    def __init__(self, 
                 num_heads:int,
                 hidden_size:int,
                 dropout_rate: float,
                 use_heads:bool = False,
                 alpha:float=0.0,
                 num_linera_heads:int=1):
        super(EncoderLayer, self).__init__()
        assert hidden_size % num_heads == 0,"hidden size should be div by num head"
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.attn_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        self.attn = MultiHeadSelfAttention(num_heads, hidden_size, dropout_rate, use_heads=use_heads, alpha=alpha, num_linear_heas=num_linera_heads)
        # self.attn = torch.nn.MultiheadAttention(hidden_size, num_heads, dropout_rate)

        self.ffn_layer_norm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        self.ffn = NormalFeedForward(hidden_size, dropout_rate, use_heads, num_heads=num_linera_heads)

    def forward(self, seqs, attn_mask=None, causal_mask=None):
        # seqs = torch.transpose(seqs, 0, 1)
        Q = self.attn_layer_norm(seqs)
        mha_outputs, _ = self.attn(Q, seqs, seqs, attn_mask=attn_mask, causal_mask=causal_mask)
        seqs = Q + mha_outputs
        # seqs = torch.transpose(seqs, 0, 1)

        seqs = self.ffn_layer_norm(seqs)
        seqs = self.ffn(seqs)
        return seqs, attn_mask

            


class CauseFormer(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CauseFormer, self).__init__()

        assert args.num_blocks % 2 == 0,"layer should be even"

        self.user_num = user_num
        self.item_num = item_num
        self.use_causal = args.use_causal
        self.use_heads = args.use_heads
        self.dev = args.device
        self.use_causal_filter = args.use_causal_filter

        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        if self.use_causal_filter:
            self.pos_emb = torch.nn.Embedding(args.maxlen + 2, args.hidden_units, padding_idx=0)
        else:
            self.pos_emb = torch.nn.Embedding(args.maxlen + 1, args.hidden_units, padding_idx=0)
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        self.pq_m = 8
        # self.item_code = ItemCode(self.pq_m, args.hidden_units, item_num, args.maxlen, args.device)

        if args.use_causal:
            self.causal_layer = torch.nn.ModuleList([EncoderLayer(args.num_heads, args.hidden_units, args.dropout_rate, args.use_heads, args.alpha, args.num_linera_heads) for _ in range(args.num_blocks // 2)])
            self.decoder_layer = torch.nn.ModuleList([EncoderLayer(args.num_heads, args.hidden_units, args.dropout_rate, args.use_heads, args.alpha, args.num_linera_heads) for _ in range(args.num_blocks // 2)])
        else:
            self.decoder_layer = torch.nn.ModuleList([EncoderLayer(args.num_heads, args.hidden_units, args.dropout_rate, args.use_heads, args.alpha, args.num_linera_heads) for _ in range(args.num_blocks)])
        
        if args.use_causal_filter:
            self.output = torch.nn.Linear(args.hidden_units, 1)

    def batch_cov(self, points):
        B, N, D = points.size()
        mean = points.mean(dim=1).unsqueeze(1)
        diffs = (points - mean).reshape(B * N, D)
        prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
        bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
        return bcov  # (B, D, D)

    # def log2feats(
    #     self, log_seqs, target=None, 
    # ):  
    #     # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
    #     seqs = self.item_code(torch.LongTensor(log_seqs).to(self.dev))  # (256, 200) -> (256, 200, 48)
    #     # seqs *= self.item_emb.embedding_dim**0.5 # scaling?
    #     poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
    #     # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
    #     poss *= log_seqs != 0
    #     seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
    #     seqs = self.emb_dropout(seqs)

    #     tl = seqs.shape[1]  # time dim len for enforce causality
    #     attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     if self.use_causal:
    #         if not self.use_causal_filter:
    #             for layer in self.causal_layer:
    #                 causal_z, attention_mask = layer(seqs, attention_mask)
    #             causal_weight = self.batch_cov(causal_z.permute(0, 2, 1)).clamp(max=3)
    #             # causal_weight *= torch.tril(torch.ones((tl, tl), dtype=torch.float, device=self.dev)).unsqueeze(0)
    #             causal_mask = torch.where(causal_weight > 0.7, 0.0, 1.0)
    #         else:
    #             target
        
    #     else:
    #         causal_weight = None
    #         causal_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

    #     for layer in self.decoder_layer:
    #         seqs,causal_mask = layer(seqs, causal_mask)

    #     log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

    #     return log_feats, causal_weight


    # def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs=None):  # for training
    #     log_feats, causal_mask = self.log2feats(log_seqs)  # user_ids hasn't been used yet

    #     pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
    #     if neg_seqs is not None:
    #         neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

    #     pos_logits = (log_feats * pos_embs).sum(dim=-1)
    #     if neg_seqs is not None:
    #         neg_logits = (log_feats * neg_embs).sum(dim=-1)
    #     else:
    #         neg_logits = None

    #     return pos_logits, neg_logits, causal_mask  # pos_pred, neg_pred

    def log2feats(
        self, log_seqs, target=None, 
    ):  
        # TODO: fp64 and int64 as default in python, trim? Use Transformer get sequence feature?
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        # seqs *= self.item_emb.embedding_dim**0.5 # scaling?
        poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
        # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
        poss *= log_seqs != 0
        seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        if target is not None:
            tensor_target_seqs = torch.LongTensor(target).to(self.dev)
            tensor_log_seqs = torch.LongTensor(log_seqs).to(self.dev)
            tensor_causal_seqs = torch.cat([tensor_log_seqs,tensor_target_seqs[:, -1].unsqueeze(1)], dim=-1)
            causal_seqs = self.item_emb(tensor_causal_seqs)
            causal_seqs *= self.item_emb.embedding_dim ** 0.5
            # seqs *= self.item_emb.embedding_dim**0.5 # scaling?
            poss = np.tile(np.arange(1, tensor_causal_seqs.shape[1] + 1), [tensor_causal_seqs.shape[0], 1])
            # TODO: directly do tensor = torch.arange(1, xxx, device='cuda') to save extra overheads
            poss *= tensor_causal_seqs.cpu().numpy() != 0
            causal_seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
            causal_seqs = self.emb_dropout(causal_seqs)
            causal_attn_mask_tl = causal_seqs.shape[1]
            causal_attn_mask = ~torch.tril(torch.ones((causal_attn_mask_tl, causal_attn_mask_tl), dtype=torch.bool, device=self.dev))
        else:
            causal_seqs = seqs
            
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        

        if self.use_causal:
            if not self.use_causal_filter:
                for layer in self.causal_layer:
                    causal_z, _ = layer(seqs, attn_mask=attention_mask)
                causal_weight = self.batch_cov(causal_z.permute(0, 2, 1)).clamp(max=3)
                # causal_weight *= torch.tril(torch.ones((tl, tl), dtype=torch.float, device=self.dev)).unsqueeze(0)
                # causal_mask = torch.where(causal_weight < 0.9, 0.0, causal_weight)
                # causal_mask = torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
                causal_mask = causal_weight
            else:
                causal_z = causal_seqs
                for layer in self.causal_layer:
                    causal_z, _ = layer(causal_z, attn_mask=causal_attn_mask)
                assert not torch.any(torch.isnan(causal_seqs)),f"{causal_seqs}"
                assert not torch.any(torch.isinf(causal_seqs)),f"{causal_seqs}"
                # causal_weight = self.batch_cov(causal_z.permute(0, 2, 1)).clamp(max=3)
                causal_weight = torch.matmul(causal_z, causal_z.permute(0, 2, 1))
                # causal_mask = torch.where(causal_weight > 0.7, 0.0, 1.0)
                causal_mask = causal_weight[:, 1:, :-1]
                assert not torch.any(torch.isnan(causal_weight)),f"{causal_weight} {causal_z}"
                assert not torch.any(torch.isinf(causal_weight)),f"{causal_weight}"
                diag = torch.diag_embed(1.0 / torch.sqrt(torch.diagonal(causal_weight,dim1=-2,dim2=-1)),dim1=1,dim2=2)
                assert not torch.any(torch.isnan(diag)),f"{diag},{torch.diagonal(causal_weight,dim1=-2,dim2=-1)}" 
                assert not torch.any(torch.isinf(diag)),f"{diag},{torch.diagonal(causal_weight,dim1=-2,dim2=-1)}" 
                causal_weight = torch.matmul(torch.matmul(diag,causal_weight),diag)
                assert not torch.any(torch.isnan(causal_weight)),f"{causal_weight}"
                assert not torch.any(torch.isinf(causal_weight)),f"{causal_weight}"
        
        else:
            causal_weight = None
            # causal_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
            causal_mask = None

        for layer in self.decoder_layer:
            seqs,causal_mask = layer(seqs, attn_mask=attention_mask, causal_mask=causal_mask)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats, causal_weight

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs=None):  # for training
        if self.use_causal_filter:
            pos_log_feats, pos_causal_mask = self.log2feats(log_seqs, pos_seqs)  # user_ids hasn't been used yet
            neg_log_feats, neg_causal_mask = self.log2feats(log_seqs, neg_seqs)

            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            if neg_seqs is not None:
                neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (pos_log_feats * pos_embs).sum(dim=-1)
            # pos_logits = self.output(pos_log_feats).squeeze()
            if neg_seqs is not None:
                neg_logits = (neg_log_feats * neg_embs).sum(dim=-1)
                # neg_logits = self.output(neg_log_feats).squeeze()
            else:
                neg_logits = None
            
            return pos_logits, neg_logits, pos_causal_mask  # pos_pred, neg_pred
        else:
            log_feats, causal_mask = self.log2feats(log_seqs)  # user_ids hasn't been used yet

            pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
            if neg_seqs is not None:
                neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

            pos_logits = (log_feats * pos_embs).sum(dim=-1)
            if neg_seqs is not None:
                neg_logits = (log_feats * neg_embs).sum(dim=-1)
            else:
                neg_logits = None
            
            return pos_logits, neg_logits, causal_mask  # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices):  # for inference
        tensor_log_seqs = torch.LongTensor(log_seqs)
        tensor_item_indices = torch.LongTensor(item_indices)
        if not self.use_causal_filter:
            log_feats, causal_mask = self.log2feats(log_seqs)  # user_ids hasn't been used yet

            final_feat = log_feats[:, -1, :]  # only use last QKV classifier, a waste

            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

            # preds = self.pos_sigmoid(logits) # rank same item list for different users

            return logits  # preds # (U, I)
        else:
            log_feats, causal_mask = self.log2feats(tensor_log_seqs.repeat([len(item_indices),1]).numpy(), tensor_item_indices.view([-1,1]).numpy())

            final_feat = log_feats[:, -1, :]

            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev))  # (U, I, C)

            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
            # logits = self.output(final_feat).permute(1,0)

            return logits






    def get_seq_embedding(self, input_ids):
        pass