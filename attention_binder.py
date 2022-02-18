
# corrected sample code for
# "Deciphering antibody affinity maturation with language models and weakly supervised learning"

import torch


# hyper-parameters

L = 30  # input sequence length
d_hid = 512  # encoder output dimension

d_emb = 32  # instance embedding dimension
n_heads = 4  # instance attention
d_head = 16  # instance attention
d_attn = d_emb  # bag feature dimension, TODO: check if that is correct

act_fn = torch.nn.Identity()  # TODO: check where to apply non-linearities (which ?) and adding biases to transforms


# MIL input

B = 8  # bag size, either positive or negative sequences
S = torch.randn((B, L, d_hid))  # output of the pre-trained transformer encoder, e.g. BERT-like
pad_pos = torch.randint(2, L, (B, 1))  # positions from which padding token start
pad_mask = torch.arange(L).unsqueeze(0).repeat(B, 1)  # corresponding pad token mask
pad_mask = torch.where(pad_mask >= pad_pos.repeat(1, L), torch.ones_like(pad_mask), torch.zeros_like(pad_mask))


#########
# forward in the instance embedding

# weights

W_h_e = torch.randn((d_emb, d_hid))  # input layer to instance embedding

QKV_emb = torch.randn((n_heads*d_head*3, d_emb))
W_emb = [torch.randn((1, d_head)) for i_head in range(n_heads)]  # are separate for each head logits

W_x = torch.randn((d_attn, n_heads*d_head))  # output layer between instance embedding and bag classifier

# transformations

# output of first attention should be of shape a_emb = B,L,1
# by concatenating them we get per residue attention of shape B,n_heads,L,1
# --> a_emb used to annotate the antibody structure
# and by pooling with per residue value of shape B,n_heads,L,d_head
# we get bag of embeddings of shape B,n_heads*d_head
# before bag classifier, we project to B, d_attn

H = torch.matmul(S, W_h_e.T)  # B,L,d_emb

proj_emb = torch.matmul(H, QKV_emb.T)  # B,L,n_heads*d_head*3
proj_emb = proj_emb.reshape(B, L, n_heads, 3*d_head)
proj_emb = proj_emb.permute(0, 2, 1, 3)  # B,n_heads,L,3*d_head
q_emb, k_emb, v_emb = proj_emb.chunk(3, dim=-1)  # B,n_heads,L,d_head

attn_logits_emb = q_emb*k_emb  # B,n_heads,L,d_head
per_residue_attn_logits = []
for i_head in range(n_heads):  # TODO: can we speed this up without for loop ?
    per_residue_attn_logits.append(torch.matmul(attn_logits_emb[:, i_head, :, :], W_emb[i_head].T))  # B,L,1
per_residue_attn_logits = torch.stack(per_residue_attn_logits, dim=1)  # B,n_heads,L,1
pad_mask = pad_mask.unsqueeze(1).repeat(1, n_heads, 1).unsqueeze(-1)
per_residue_attn_logits_masked = per_residue_attn_logits.masked_fill(pad_mask == 1, -9e15)  # -inf logits on pad positions
per_residue_attn = torch.softmax(per_residue_attn_logits_masked, dim=2)  # should be the multi-head a_emb

x = per_residue_attn*v_emb  # B,n_heads,L,d_head
x = torch.sum(x, dim=2).reshape(B, n_heads*d_head)  # B,n_heads*d_head
x = act_fn(torch.matmul(x, W_x.T))  # B,d_attn


#########
# forward in the bag classification

# weights

V_bag, U_bag = torch.randn((d_attn)), torch.randn((d_attn))
W_bag = torch.randn((1, d_attn))

W_classif = torch.randn((1, d_attn))

# transformations

attn_logits_bag = torch.sigmoid(V_bag*x) * torch.sigmoid(U_bag*x)  # B,d_attn
per_sequence_attn_logits = torch.matmul(attn_logits_bag, W_bag.T)  # B,1
per_sequence_attn = torch.softmax(per_sequence_attn_logits, dim=0)  # should be a_bag

z = per_sequence_attn*x  # B,d_attn
z = torch.sum(z, dim=0)  # d_attn

p_binder = torch.sigmoid(torch.matmul(z, W_classif.T))  # scalar prediction for the whole bag
