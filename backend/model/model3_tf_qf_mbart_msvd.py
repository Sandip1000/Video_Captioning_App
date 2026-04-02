import torch
import torch.nn as nn
from transformers import MBart50TokenizerFast
from transformers import TimesformerModel, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput





# --------------------------------------------------
# Q-Former Block
# --------------------------------------------------
class QFormerBlock(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=8, mlp_ratio=4.0):
        super().__init__()

        # Cross-attention (Queries attend to video features)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,

            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, queries, video_features):
        # Cross-attention
        cross_out, _ = self.cross_attn(
            query=queries,
            key=video_features,
            value=video_features,
        )
        queries = self.norm1(queries + cross_out)

        # Self-attention
        self_out, _ = self.self_attn(
            query=queries,
            key=queries,
            value=queries,
        )
        queries = self.norm2(queries + self_out)

        # MLP
        mlp_out = self.mlp(queries)
        queries = self.norm3(queries + mlp_out)

        return queries


# --------------------------------------------------
# Video Captioning Model with Q-Former Bridge
# --------------------------------------------------
class VideoCaptioningModel(nn.Module):
    def __init__(
        self,
        timesformer_model_name="facebook/timesformer-base-finetuned-k600",
        mbart_model_name="facebook/mbart-large-50",
        num_query_tokens=32,
        qformer_layers=4,
        freeze_timesformer=True,
        freeze_mbart_encoder=True,
    ):
        super().__init__()

        # -------- Encoder: TimeSformer --------
        self.encoder = TimesformerModel.from_pretrained(timesformer_model_name)

        # -------- Decoder: mBART --------
        self.decoder = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

        ts_hidden = self.encoder.config.hidden_size        # usually 768
        mbart_hidden = self.decoder.config.d_model         # 1024

        # -------- Projection (768 → 1024) --------
        self.video_proj = nn.Linear(ts_hidden, mbart_hidden)

        # -------- Q-Former --------
        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, mbart_hidden)
        )

        self.qformer_blocks = nn.ModuleList(
            [QFormerBlock(hidden_dim=mbart_hidden) for _ in range(qformer_layers)]
        )

        # -------- Freeze modules --------
        if freeze_timesformer:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if freeze_mbart_encoder:
            for p in self.decoder.model.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.model.decoder.parameters():
                p.requires_grad = True

    # --------------------------------------------------
    # Forward (Training)
    # --------------------------------------------------
    def forward(self, pixel_values, input_ids, labels):

        # 1️⃣ TimeSformer Encoding
        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            return_dict=True
        )

        video_features = encoder_outputs.last_hidden_state  # (B, N, 768)

        # 2️⃣ Project to 1024
        video_features = self.video_proj(video_features)

        B = video_features.size(0)

        # 3️⃣ Expand Query Tokens
        queries = self.query_tokens.expand(B, -1, -1)

        # 4️⃣ Q-Former Processing
        for block in self.qformer_blocks:
            queries = block(queries, video_features)

        # queries shape: (B, num_query_tokens, 1024)

        # 5️⃣ Wrap as Encoder Memory for mBART
        encoder_outputs = BaseModelOutput(
            last_hidden_state=queries
        )
        
        encoder_attention_mask = torch.ones(
        queries.size()[:-1],
        dtype=torch.long,
        device=queries.device
        )

        # 6️⃣ mBART Decoder
        outputs = self.decoder(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            input_ids=input_ids,
            labels=labels,
            return_dict=True
        )

        return outputs

    # --------------------------------------------------
    # Generate (Inference)
    # --------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pixel_values,
        max_length=50,
        num_beams=4,
        forced_bos_token_id=None,
        decoder_start_token_id=None,
        **generate_kwargs
    ):

        encoder_outputs = self.encoder(
            pixel_values=pixel_values,
            return_dict=True
        )

        video_features = self.video_proj(
            encoder_outputs.last_hidden_state
        )

        B = video_features.size(0)
        queries = self.query_tokens.expand(B, -1, -1)

        for block in self.qformer_blocks:
            queries = block(queries, video_features)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=queries
        )

        encoder_attention_mask = torch.ones(
        queries.size()[:-1],
        dtype=torch.long,
        device=queries.device
       ) 

        generated_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_bos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **generate_kwargs
        )

        return generated_ids
    




# Example placeholder
class VideoCaptionModel:
    def __init__(self):
        

        self.checkpoint = torch.load(fr"D:\Model_for_DEMO\tf_qf_mbart_msvd\checkpoint-6000\pytorch_model.bin")
        
        # Model Loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VideoCaptioningModel()
        self.model.load_state_dict(self.checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang= "ne_NP", tgt_lang="ne_NP" )
        

    def generate_caption(self, video_tensor):
        """Generates caption for a single video tensor"""
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # add batch dim [1, T, 3, 224, 224]
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=video_tensor,
                max_length=50,
                num_beams= 4,  
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(["ne_NP"]),
                num_return_sequences= 4,
                no_repeat_ngram_size=3,
                early_stopping=True,
                repetition_penalty=1.3
                # top_k = 50,
                # top_p = 0.9,
                # do_sample = True,
                # temperature = 1
            )
            captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return captions

# model3 = VideoCaptionModel()



