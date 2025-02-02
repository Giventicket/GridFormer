import torch
import torch.nn as nn
import math
import copy

"""
code reference: http://nlp.seas.harvard.edu/annotated-transformer/
"""

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, encoder_pe, tgt_embed, decoder_pe, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.encoder_pe = encoder_pe
        self.decoder_pe = decoder_pe
        self.generator = generator

    def forward(self, src, tgt, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src), tgt, tgt_mask)

    def encode(self, src):
        """
        src: [B, node_size, 2], no need for src_mask
        """
        src_embeddings = self.src_embed(src)
        if self.encoder_pe is not None:
            src_embeddings = self.encoder_pe(src_embeddings, src)
        return self.encoder(src_embeddings)

    def decode(self, memory, tgt, tgt_mask):
        tgt_embeddings = self.tgt_embed(tgt)
        if self.decoder_pe is not None:
            if self.decoder_pe._get_name() == "PositionalEncoding_2D":
                tgt_embeddings = self.decoder_pe(tgt_embeddings, tgt)
            else:
                tgt_embeddings = self.decoder_pe(tgt_embeddings)
            
        return self.decoder(tgt_embeddings, memory, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, sz):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, sz)

    def forward(self, x, visited_mask=None):
        logits = self.proj(x)
        if visited_mask is not None:
            logits = logits.float()
            logits = logits.masked_fill(visited_mask, -1e9)
        return nn.functional.log_softmax(logits, dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)

        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.float()
        scores = scores.masked_fill(~mask, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class PositionalEncoding_2D(nn.Module):
    "Implement the 2D PE function."

    def __init__(self, d_model, T, dropout, grid_size):
        super(PositionalEncoding_2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.grid_size = grid_size

        # Compute the div_term once.
        div_term = 1 / torch.pow(
            T, (2.0 * (torch.arange(d_model // 2))) / torch.tensor(d_model)
        )  # [batch_size, node_size, 64]
        self.register_buffer("div_term", div_term)

    def forward(self, embeddings, grid_indices):
        """
        embeddings: [batch_size, node_size, 128]
        graph: [batch_size, node_size, 2]
        """
        batch_size, node_size = grid_indices.shape
        device = grid_indices.device 
        pe_x = torch.zeros(batch_size, node_size, self.d_model // 2, device = device)  # [batch_size, node_size, 64]
        pe_y = torch.zeros(batch_size, node_size, self.d_model // 2, device = device)  # [batch_size, node_size, 64]
        
        x_term = ((grid_indices % self.grid_size) / self.grid_size + 1 / (2 * self.grid_size)).unsqueeze(-1) * self.div_term.repeat(
            batch_size, node_size, 1
        )  # [batch_size, node_size, 64]
        y_term = ((grid_indices // self.grid_size) / self.grid_size + 1 / (2 * self.grid_size)).unsqueeze(-1) * self.div_term.repeat(
            batch_size, node_size, 1
        )  # [batch_size, node_size, 64]
        
        pe_x[:, :, 0::2] = torch.sin(x_term[:, :, 0::2])  # [batch_size, node_size, 32]
        pe_x[:, :, 1::2] = torch.cos(x_term[:, :, 1::2])  # [batch_size, node_size, 32]
        pe_y[:, :, 0::2] = torch.sin(y_term[:, :, 0::2])  # [batch_size, node_size, 32]
        pe_y[:, :, 1::2] = torch.cos(y_term[:, :, 1::2])  # [batch_size, node_size, 32]
        pe = torch.cat([pe_x, pe_y], -1).requires_grad_(False)  # [batch_size, node_size, 128]
        embeddings = embeddings + pe  # [batch_size, node_size, 128]
        return self.dropout(embeddings)  # [batch_size, node_size, 128]

class CircularPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, node_size):
        super(CircularPositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(node_size, d_model)
        position = torch.arange(0, node_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term + 2 * torch.pi * position / node_size)
        pe[:, 1::2] = torch.cos(position * div_term + 2 * torch.pi * position / node_size)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
class LearnableCircularPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, node_size):
        super(LearnableCircularPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(node_size, d_model), requires_grad=True)
        position = torch.arange(0, node_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        with torch.no_grad():
            self.pe[:, 0::2] = torch.sin(position * div_term + 2 * torch.pi * position / node_size)
            self.pe[:, 1::2] = torch.cos(position * div_term + 2 * torch.pi * position / node_size)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

    
class OriginalPositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(OriginalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.pe = nn.Parameter(torch.zeros(max_len, d_model), requires_grad=True)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        with torch.no_grad():
            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

def make_model(
    grid_size, 
    N, 
    d_model,
    d_ff, 
    h, 
    dropout,
    encoder_pe_option,
    decoder_pe_option,
    use_start_token,
    share_lut,
    node_size
    ):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    src_vocab = grid_size * grid_size
    if use_start_token:
        tgt_vocab = grid_size * grid_size + 1
    else:
        tgt_vocab = grid_size * grid_size 

    if encoder_pe_option == "pe_2d":
        encoder_pe = PositionalEncoding_2D(d_model, 2, dropout, grid_size)
    else:
        encoder_pe = None
        
    if decoder_pe_option == "pe_2d":
        decoder_pe = PositionalEncoding_2D(d_model, 2, dropout, grid_size)
    elif decoder_pe_option == "pe_1d_original":
        decoder_pe = OriginalPositionalEncoding(d_model, dropout, 10000)
    elif decoder_pe_option == "pe_1d_learnable":
        decoder_pe = LearnablePositionalEncoding(d_model, dropout, 10000)
    elif decoder_pe_option == "pe_1d_circular":
        decoder_pe = CircularPositionalEncoding(d_model, dropout, node_size)
    elif decoder_pe_option == "pe_1d_learnable_circular":
        decoder_pe = LearnableCircularPositionalEncoding(d_model, dropout, node_size)
    else:
        decoder_pe = None
        
    if share_lut:
        tgt_embed = src_embed = Embeddings(d_model, tgt_vocab)
    else:
        tgt_embed = Embeddings(d_model, src_vocab) 
        src_embed = Embeddings(d_model, tgt_vocab)

    model = EncoderDecoder(
        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed = src_embed,
        encoder_pe = encoder_pe,
        tgt_embed = tgt_embed,
        decoder_pe = decoder_pe,
        generator = Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
