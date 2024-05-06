import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, alpha=1.0):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha

        # Define the low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(output_dim, rank))
        self.B = nn.Parameter(torch.randn(rank, input_dim))

    def forward(self, W):
        # Compute the low-rank update
        delta_W = self.alpha * self.A @ self.B
        # Return the adapted weights
        return W + delta_W


class LoRATransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, rank, alpha=1.0, batch_first=True):
        super(LoRATransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=batch_first)
        
        # Initialize LoRA for the key and value projections in the attention mechanism
        self.lora_attn_key = LoRALayer(d_model, d_model, rank, alpha)
        self.lora_attn_value = LoRALayer(d_model, d_model, rank, alpha)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Apply LoRA updates
        original_key_weight = self.self_attn.in_proj_weight[:self.self_attn.embed_dim, :]
        original_value_weight = self.self_attn.in_proj_weight[self.self_attn.embed_dim:2*self.self_attn.embed_dim, :]

        adapted_key_weight = self.lora_attn_key(original_key_weight)
        adapted_value_weight = self.lora_attn_value(original_value_weight)

        # Temporarily replace the weights
        self.self_attn.in_proj_weight.data[:self.self_attn.embed_dim, :] = adapted_key_weight
        self.self_attn.in_proj_weight.data[self.self_attn.embed_dim:2*self.self_attn.embed_dim, :] = adapted_value_weight

        # Perform the attention operation
        output, attn_output_weights = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)

        # Restore original weights (optional if weights are not shared)
        self.self_attn.in_proj_weight.data[:self.self_attn.embed_dim, :] = original_key_weight
        self.self_attn.in_proj_weight.data[self.self_attn.embed_dim:2*self.self_attn.embed_dim, :] = original_value_weight

        return output
    
class TransformerModel_multiclass_LoRA(nn.Module):
    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 classes: int = 4, rank: int = 10, alpha: float = 1.0):
        super(TransformerModel_multiclass_LoRA, self).__init__()
        self.model_type = 'Transformer'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = nn.LayerNorm(d_model)
        # define the encoder layers
        encoder_layers = LoRATransformerEncoderLayer(d_model, nhead, rank, alpha, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model*seq_len, d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """
        #src = src * math.sqrt(self.d_model)
        src = self.norm(src)
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output
    
class TransformerModel_multiclass_v2_LoRA(nn.Module):

    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 dropout: float = 0.1, classes: int = 4, use_pos: bool = False, rank: int = 10, alpha: float = 1.0):
        super(TransformerModel_multiclass_v2_LoRA, self).__init__()
        self.model_type = 'Transformer'

        self.norm = nn.LayerNorm(d_model)
        # create the positional encoder
        self.use_positional_enc = use_pos
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        # define [CLS] token to be used for classification
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, d_model))
        # define the encoder layers
        encoder_layers = LoRATransformerEncoderLayer(d_model, nhead, rank, alpha, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # we will not use the decoder
        # instead we will add a linear layer, another scaled dropout layer, and finally a classifier layer
        self.pre_classifier = torch.nn.Linear(d_model, d_model)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
        Returns:
            output classifier label
        """

        # We normalize the input weights 
        cls_tokens = self.cls_token.repeat(src.size(0),1,1)
        src = torch.column_stack((cls_tokens, src))
        src = self.norm(src)
        if self.use_positional_enc:
            src = self.pos_encoder(src).squeeze()
        t_out = self.transformer_encoder(src)
        # get hidden state of the [CLS] token
        t_out = t_out[:,0,:].squeeze()
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output
    
class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.prompt, dim=1)
            self.prompt_key = prompt_mean
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None):
        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(self.prompt_key, dim=1) # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k

            batched_prompt_raw = self.prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            # out['prompt_norm'] = prompt_norm
            # out['x_embed_norm'] = x_embed_norm
            # out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        return out

class TransformerModel_multiclass_with_Prompts(nn.Module):
    def __init__(self, d_model: int = 512, seq_len: int = 64, nhead: int = 8, nlayers: int = 2,
                 classes: int = 4, num_prompt_tokens: int = 10, embed_dim: int = 512):
        super(TransformerModel_multiclass_with_Prompts, self).__init__()
        self.model_type = 'Transformer'

        # Initialize the prompt module
        self.prompt_module = Prompt(length=num_prompt_tokens, embed_dim=embed_dim, prompt_pool=True, pool_size=100, top_k=5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.norm = nn.LayerNorm(d_model)
        
        # Define the encoder layers
        encoder_layers = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

        # Adjust sequence length for prompt tokens dynamically based on prompt length
        self.pre_classifier = torch.nn.Linear(d_model * (seq_len + num_prompt_tokens), d_model)
        self.dropout = torch.nn.Dropout(0.5)
        self.classifier = torch.nn.Linear(d_model, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src: Tensor, cls_features: Tensor = None) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, features]
            cls_features: Optional Tensor for class-specific features
        Returns:
            output classifier label
        """
        # Process prompts
        prompt_output = self.prompt_module(src, cls_features=cls_features)
        src = prompt_output['prompted_embedding']  # Get the modified embeddings with prompts

        # Normalize and encode
        src = self.norm(src)
        t_out = self.transformer_encoder(src)
        t_out = torch.flatten(t_out, start_dim=1)
        
        # Classifier steps
        pooler = self.pre_classifier(t_out)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        output = self.sigmoid(output)
        return output


