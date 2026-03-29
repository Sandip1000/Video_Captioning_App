import torch
import torch.nn as nn
from transformers import MBart50TokenizerFast
from transformers import TimesformerModel, MBartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from safetensors.torch import load_file


# class VideoCaptioningModel(nn.Module):
#     def __init__(self,
#         timesformer_model_name="facebook/timesformer-base-finetuned-k600",
#         mbart_model_name="facebook/mbart-large-50",
#         freeze_timesformer=True,
#     ):
#         super().__init__()

#         # -------- Encoder: TimeSformer --------
#         self.encoder = TimesformerModel.from_pretrained(timesformer_model_name)

#         # -------- Decoder: mBART (encoder unused) --------
#         self.decoder = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

#         # -------- Dimensions --------
#         ts_hidden = self.encoder.config.hidden_size          # e.g. 768
#         mbart_hidden = self.decoder.config.d_model           # e.g. 1024

#         # -------- Adapter (TimeSformer → mBART) --------
#         self.adapter = nn.Sequential(
#             nn.LayerNorm(ts_hidden),
#             nn.Linear(ts_hidden, mbart_hidden),
#             nn.GELU(),
#             nn.LayerNorm(mbart_hidden)
#         )
        
#         nn.init.normal_(self.adapter[1].weight, mean=0, std=0.02)
#         nn.init.zeros_(self.adapter[1].bias)

        
#         # Optional: freeze TimeSformer
#         if freeze_timesformer:
#             for p in self.encoder.parameters():
#                 p.requires_grad = False
        


#         for p in self.decoder.model.encoder.parameters():
#             p.requires_grad = False

            
#     def forward(
#         self,
#         pixel_values,      # (B, T, C, H, W)
#         input_ids,
#         labels
        
#     ):
#         """
#         pixel_values : video tensor
#         input_ids    : decoder tokens
#         labels       : for training (optional)
#         """

#         # -------- 1. TimeSformer Encoding --------
#         encoder_outputs = self.encoder(
#             pixel_values=pixel_values,
#             return_dict=True
#         )

#         # (B, seq_len, ts_hidden)
#         video_features = encoder_outputs.last_hidden_state

#         # -------- 2. Adapter Projection --------
#         adapted_features = self.adapter(video_features)
#         # (B, seq_len, mbart_hidden)

#         # -------- 3. Wrap in BaseModelOutput --------
#         encoder_outputs = BaseModelOutput(
#             last_hidden_state=adapted_features
#         )

#         # -------- 4. mBART Decoder with Cross-Attention --------
#         outputs = self.decoder(
#             encoder_outputs = encoder_outputs,  # <-- cross-attention source
#             input_ids = input_ids,
#             labels = labels,
#             return_dict = True
#         )

#         return outputs

#     @torch.no_grad()
#     def generate(
#         self,
#         pixel_values,             # video input
#         max_length=50,
#         num_beams=4,
#         forced_bos_token_id=None,
#         decoder_start_token_id=None,
#         **generate_kwargs
#     ):
#         """
#         Generates captions for given video frames.
#         """
#         self.eval()

#         # 1. TimeSformer Encoding
#         encoder_outputs = self.encoder(
#             pixel_values=pixel_values,
#             return_dict=True
#         )
#         video_features = encoder_outputs.last_hidden_state

#         # 2. Adapter projection
#         adapted_features = self.adapter(video_features)

#         # 3. Wrap in BaseModelOutput
#         encoder_outputs = BaseModelOutput(
#             last_hidden_state=adapted_features
#         )

#         # 4. Generate with mBART
#         generated_ids = self.decoder.generate(
#             encoder_outputs=encoder_outputs,
#             max_length=max_length,
#             num_beams=num_beams,
#             forced_bos_token_id=forced_bos_token_id,
#             decoder_start_token_id=decoder_start_token_id,
#             **generate_kwargs
#         )

#         return generated_ids












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






# class QFormerBlock(nn.Module):
#     def __init__(self, hidden_dim=1024, num_heads=8, mlp_ratio=4.0, dropout=0.1):
#         super().__init__()

#         # Cross-attention (Queries attend to video features)
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True,
#             dropout=dropout  # dropout inside attention
#         )
#         self.norm1 = nn.LayerNorm(hidden_dim)
#         self.cross_dropout = nn.Dropout(dropout)  # after residual

#         # Self-attention among queries
#         self.self_attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True,
#             dropout=dropout
#         )
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.self_dropout = nn.Dropout(dropout)

#         # Feed-forward
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
#             nn.GELU(),
#             nn.Dropout(dropout),  # MLP dropout
#             nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
#         )
#         self.norm3 = nn.LayerNorm(hidden_dim)
#         self.mlp_dropout = nn.Dropout(dropout)

#     def forward(self, queries, video_features):
#         # Cross-attention
#         cross_out, _ = self.cross_attn(
#             query=queries,
#             key=video_features,
#             value=video_features,
#         )
#         queries = self.norm1(queries + self.cross_dropout(cross_out))

#         # Self-attention
#         self_out, _ = self.self_attn(
#             query=queries,
#             key=queries,
#             value=queries,
#         )
#         queries = self.norm2(queries + self.self_dropout(self_out))

#         # MLP
#         mlp_out = self.mlp(queries)
#         queries = self.norm3(queries + self.mlp_dropout(mlp_out))

#         return queries

# # --------------------------------------------------
# # Video Captioning Model with Q-Former Bridge
# # --------------------------------------------------
# class VideoCaptioningModel(nn.Module):
#     def __init__(
#         self,
#         timesformer_model_name="facebook/timesformer-base-finetuned-k600",
#         mbart_model_name="facebook/mbart-large-50",
#         num_query_tokens=32,
#         qformer_layers=4,
#         freeze_timesformer=True,
#         freeze_mbart_encoder=True,
#     ):
#         super().__init__()

#         # -------- Encoder: TimeSformer --------
#         self.encoder = TimesformerModel.from_pretrained(timesformer_model_name)

#         # -------- Decoder: mBART --------
#         self.decoder = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

#         ts_hidden = self.encoder.config.hidden_size        # usually 768
#         mbart_hidden = self.decoder.config.d_model         # 1024

#         # -------- Projection (768 → 1024) --------
#         self.video_proj = nn.Linear(ts_hidden, mbart_hidden)

#         # -------- Q-Former --------
#         self.query_tokens = nn.Parameter(
#             torch.randn(1, num_query_tokens, mbart_hidden)
#         )

#         self.qformer_blocks = nn.ModuleList(
#             [QFormerBlock(hidden_dim=mbart_hidden) for _ in range(qformer_layers)]
#         )

#         # Qformer_total_parameters = sum(p.numel() for p in self.qformer_blocks.parameters())
#         # Qformer_trainable_parameters = sum(p.numel() for p in self.qformer_blocks.parameters() if p.requires_grad)
#         # print("Qformer Total parameters: ", Qformer_total_parameters)
#         # print("Qformer Trainable parameters: ", Qformer_trainable_parameters)
#         # print("Qformer Non-Trainable parameters: ", Qformer_total_parameters- Qformer_trainable_parameters)

#         # -------- Freeze modules --------
#         if freeze_timesformer:
#             for p in self.encoder.parameters():
#                 p.requires_grad = False
        
#         #Unfreeze last 4 TimeSformer blocks
#         for block in self.encoder.encoder.layer[-4:]:
#             for p in block.parameters():
#                 p.requires_grad = True
        
#         # timesformer_total_parameters = sum(p.numel() for p in self.encoder.parameters())
#         # timesformer_trainable_parameters = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
#         # print("Timesformer Total parameters: ", timesformer_total_parameters)
#         # print("Timesformer Trainable parameters: ", timesformer_trainable_parameters)
#         # print("Timesformer Non-Trainable parameters: ", timesformer_total_parameters- timesformer_trainable_parameters)


        
#         if freeze_mbart_encoder:
#             for p in self.decoder.model.encoder.parameters():
#                 p.requires_grad = False
#             for p in self.decoder.model.decoder.parameters():
#                 p.requires_grad = True

        
#         # mbart_total_parameters = sum(p.numel() for p in self.decoder.parameters())
#         # mbart_trainable_parameters = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
#         # print("mBART Total parameters: ", mbart_total_parameters)
#         # print("mBART Trainable parameters: ", mbart_trainable_parameters)
#         # print("mBART Non-Trainable parameters: ", mbart_total_parameters- mbart_trainable_parameters)

#     # --------------------------------------------------
#     # Forward (Training)
#     # --------------------------------------------------
#     def forward(self, pixel_values, input_ids, labels):

#         # 1️⃣ TimeSformer Encoding
#         encoder_outputs = self.encoder(
#             pixel_values=pixel_values,
#             return_dict=True
#         )

#         video_features = encoder_outputs.last_hidden_state  # (B, N, 768)

#         # 2️⃣ Project to 1024
#         video_features = self.video_proj(video_features)

#         B = video_features.size(0)

#         # 3️⃣ Expand Query Tokens
#         queries = self.query_tokens.expand(B, -1, -1)

#         # 4️⃣ Q-Former Processing
#         for block in self.qformer_blocks:
#             queries = block(queries, video_features)

#         # queries shape: (B, num_query_tokens, 1024)

#         # 5️⃣ Wrap as Encoder Memory for mBART
#         encoder_outputs = BaseModelOutput(
#             last_hidden_state=queries
#         )
        
#         encoder_attention_mask = torch.ones(
#         queries.size()[:-1],
#         dtype=torch.long,
#         device=queries.device
#         )

#         # 6️⃣ mBART Decoder
#         outputs = self.decoder(
#             encoder_outputs=encoder_outputs,
#             attention_mask=encoder_attention_mask,
#             input_ids=input_ids,
#             labels=labels,
#             return_dict=True
#         )

#         return outputs

#     # --------------------------------------------------
#     # Generate (Inference)
#     # --------------------------------------------------
#     @torch.no_grad()
#     def generate(
#         self,
#         pixel_values,
#         max_length=32,
#         num_beams=4,
#         forced_bos_token_id=None,
#         decoder_start_token_id=None,
#         **generate_kwargs
#     ):

#         encoder_outputs = self.encoder(
#             pixel_values=pixel_values,
#             return_dict=True
#         )

#         video_features = self.video_proj(
#             encoder_outputs.last_hidden_state
#         )

#         B = video_features.size(0)
#         queries = self.query_tokens.expand(B, -1, -1)

#         for block in self.qformer_blocks:
#             queries = block(queries, video_features)

#         encoder_outputs = BaseModelOutput(
#             last_hidden_state=queries
#         )

#         encoder_attention_mask = torch.ones(
#         queries.size()[:-1],
#         dtype=torch.long,
#         device=queries.device
#        ) 

#         generated_ids = self.decoder.generate(
#             encoder_outputs=encoder_outputs,
#             attention_mask=encoder_attention_mask,
#             max_length=max_length,
#             num_beams=num_beams,
#             forced_bos_token_id=forced_bos_token_id,
#             decoder_start_token_id=decoder_start_token_id,
#             **generate_kwargs
#         )

#         return generated_ids


































# Example placeholder
class VideoCaptionModel:
    def __init__(self):
        

        self.checkpoint = load_file(fr"C:\Users\sanje\Desktop\Video_Captioning_App\backend\model.safetensors")
        
        # Model Loading
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.checkpoint = torch.load(fr"C:\Users\sanje\Desktop\Video_Captioning_App\backend\checkpoint-8000\pytorch_model.bin", map_location = self.device)
        self.model = VideoCaptioningModel()
        self.model.load_state_dict(self.checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50", src_lang = "ne_NP", tgt_lang = "ne_NP")
        

    def generate_caption(self, video_tensor):
        """Generates caption for a single video tensor"""
        video_tensor = video_tensor.unsqueeze(0).to(self.device)  # add batch dim [1, T, 3, 224, 224]
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=video_tensor,
                max_length=50,
                num_beams= 4,  
                forced_bos_token_id=self.tokenizer.lang_code_to_id["ne_NP"],
                num_return_sequences= 4,
                # top_k = 50,
                # top_p = 0.9,
                # do_sample = True,
                # temperature = 1,
                # repetition_penalty=1.2,
                # no_repeat_ngram_size=2,
            )
            captions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return captions

model = VideoCaptionModel()
