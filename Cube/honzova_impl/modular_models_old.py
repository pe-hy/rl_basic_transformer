import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


class RNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.q_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        #self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        #if not self.flash:
        #    print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = 1 - torch.eye(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", bias)
        self.RNN = nn.RNN(input_size=config.n_embd, hidden_size=config.n_embd, num_layers=1, nonlinearity='tanh', bias=True, batch_first=True, dropout=config.dropout, bidirectional=False)
        # flip 0s and 1s


    def forward(self, x, hidden):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        q = self.q_attn(hidden)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, 1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #if self.flash:
            # efficient attention using Flash Attention CUDA kernels
        #    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        #else:
            # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.squeeze(1) #transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.n_embd)
        # attn_weights = F.softmax(attn_scores, dim=-1)
        # output projection
        #_,hidden = self.RNN(y, hidden)
        #hidden = self.resid_dropout(self.c_proj(hidden.squeeze(0))).unsqueeze(0)
        hidden = y.unsqueeze(0)
        hidden = self.resid_dropout(self.c_proj(hidden.squeeze(0))).unsqueeze(0)
        return x, hidden


class ModularGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.n_iters = config.n_iters
        self.init_bottleneck_by_last = config.init_bottleneck_by_last
        self.hidden = nn.Embedding(1, config.n_embd)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            rnn = RNN(config)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_iters))

        # report number of parameters
        #print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        # reinitialize the last token to random unit vector
        # tok_emb[:,-1,:] = torch.nn.init.normal_(torch.empty(tok_emb[:,-1,:].shape), mean=0.0, std=0.02).to(device)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb
        if self.init_bottleneck_by_last:
            x = x[:, :-1, :] # remove the last token
            hidden = tok_emb[:, -1, :].unsqueeze(0) # initial hidden state
        # for block in self.transformer.h: # experiment
        #     x = block(x)
        # initial hidden state by repeating the self.hidden embedding across the batch dimension
        else:
            hidden = self.hidden.weight[0, :].repeat(b,1, 1).transpose(0,1)
        layer = self.transformer.rnn
        for i in range(self.n_iters):
            x, hidden = layer(x, hidden)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            #logits = self.lm_head(x)
            logits = self.lm_head(hidden)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(hidden) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        return optimizer