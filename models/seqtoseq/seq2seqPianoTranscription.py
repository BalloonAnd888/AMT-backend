import torch
from torch import nn
from models.seqtoseq.blocks.encoderLayer import EncoderLayer
from models.seqtoseq.blocks.decoderLayer import DecoderLayer
from models.seqtoseq.embedding.positionalEncoding import PositionalEncoding
from models.seqtoseq.embedding.tokenEmbedding import TokenEmbedding

class Seq2SeqPianoTranscription(nn.Module):
    def __init__(self, input_dim, vocab_size, d_model=512, num_heads=8, d_ff=2048, 
                 num_encoder_layers=6, num_decoder_layers=6, dropout=0.1, max_seq_len=4096):
        super(Seq2SeqPianoTranscription, self).__init__()
        
        # Encoder Input Processing: maps input audio to the working dimension directly via linear transformation.
        self.input_linear = nn.Linear(input_dim, d_model)
        self.src_pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Decoder Input Processing: maps discrete token IDs to their dense equivalents.
        self.tgt_embedding = TokenEmbedding(vocab_size, d_model)
        self.tgt_pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Encoder Stack
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)
        
        # Decoder Stack
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        
        # Language Head
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def generate_square_subsequent_mask(self, sz):
        # Generates a standard causal mask to restrict attention to prior context
        mask = torch.tril(torch.ones(sz, sz))
        return mask.view(1, 1, sz, sz)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
            
        enc_output = self.input_linear(src)
        enc_output = self.src_pos_encoding(enc_output)
        
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        enc_output = self.enc_norm(enc_output)
            
        dec_output = self.tgt_embedding(tgt)
        dec_output = self.tgt_pos_encoding(dec_output)
        
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, src_mask, tgt_mask)
        dec_output = self.dec_norm(dec_output)
            
        output = self.fc_out(dec_output)
        return output
