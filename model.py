from mlx.utils import tree_unflatten
from typing import OrderedDict
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import tiktoken
from transformers import AutoModelForCausalLM


@dataclass
class GPTModelConfig:
    n_vocab = 50257
    n_pos = 1024
    n_hidden_dim = 768

    n_hidden_layers = 12
    n_attn_heads = 12

    layer_norm_eps = 1e-5

    tokenizer = tiktoken.encoding_for_model("gpt2")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()

        embed_dim = config.n_hidden_dim

        self.num_heads = config.n_attn_heads
        self.head_dim = embed_dim // self.num_heads

        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x, mask=None, cache=None):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = mx.split(qkv, 3, axis=-1)

        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            key_cache, value_cache = cache
            k = mx.concatenate([key_cache, k], axis=2)
            v = mx.concatenate([value_cache, v], axis=2)
        
        cache = (k, v)
            

        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / mx.sqrt(self.head_dim),
            mask=mask,
        )

        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)

        return self.c_proj(y), cache


class MLP(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()

        embed_dim = config.n_hidden_dim

        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.act = nn.GELU(approx="tanh")

    def __call__(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()

        embed_dim = config.n_hidden_dim
        eps = config.layer_norm_eps

        self.ln_1 = nn.LayerNorm(embed_dim, eps)
        self.ln_2 = nn.LayerNorm(embed_dim, eps)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def __call__(self, x, mask=None, cache=None):
        attn_out, cache = self.attn(self.ln_1(x), mask, cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))

        return x, cache


class GPT2(nn.Module):
    def __init__(self, config: GPTModelConfig):
        super().__init__()

        n_vocab = config.n_vocab
        n_pos = config.n_pos
        hidden_dim = config.n_hidden_dim
        n_layers = config.n_hidden_layers
        eps = config.layer_norm_eps

        self.wte = nn.Embedding(n_vocab, hidden_dim)
        self.wpe = nn.Embedding(n_pos, hidden_dim)
        self.h = [Block(config) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(hidden_dim, eps=eps)
        self.lm_head = nn.Linear(hidden_dim, n_vocab, bias=False)

        self.tokenizer = config.tokenizer

    def __call__(self, x, cache=None):
        B, T = x.shape

        past_len = cache[0][0].shape[2] if cache is not None else 0
        positions = mx.arange(past_len, past_len + T)

        x = self.wte(x) + self.wpe(positions)

        mask = None
        if T > 1:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
            mask = mask.astype(x.dtype)

        new_cache = []

        for i, block in enumerate(self.h):
            c = cache[i] if cache is not None else None
            x, c = block(x, mask=mask, cache=c)
            new_cache.append(c)

        # Final norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_cache

    def load_weights_from_hf_state_dict(self, state_dict: OrderedDict):
        mlx_weights = {}

        for k, v in state_dict.items():
            if k.startswith("transformer."):
                k = k.replace("transformer.", "")

            if any(w in k for w in ["c_attn.weight", "c_proj.weight", "c_fc.weight"]):
                mlx_weights[k] = mx.array(v).T
            else:
                mlx_weights[k] = mx.array(v)

        mlx_weights_unflattened = tree_unflatten(mlx_weights)

        self.update(mlx_weights_unflattened)
        self.lm_head.weight = self.wte.weight

    def sample_next_token(self, logits, temperature=1.0, top_k=None):
        logits = logits[:, -1, :]

        if temperature <= 0:
            return mx.argmax(logits, axis=-1, keepdims=True)

        logits = logits / temperature

        if top_k is not None:
            values = mx.topk(logits, top_k, axis=-1)
            cutoff = values[:, :1]
            logits = mx.where(logits < cutoff, -mx.inf, logits)

        return mx.random.categorical(logits, axis=-1).reshape(-1, 1)

    def predict(self, prompt, max_tokens=500, temperature=0.8, top_k=25):
        input_ids = mx.array([self.tokenizer.encode(prompt)])
        eos_token_id = self.tokenizer.eot_token

        # Pre-fill phase (process entire prompt at once to build KV cache)
        logits, cache = self(input_ids)
        next_token = self.sample_next_token(logits, temperature, top_k)

        @mx.compile
        def step(token, current_cache):
            logits, new_cache = self(token, cache=current_cache)
            next_tok = self.sample_next_token(logits, temperature, top_k)
            return next_tok, new_cache

        # Generation loop
        token_count = 0
        while token_count < max_tokens:
            tok_id = next_token.item()

            # 1. Check for EOS first so we don't print the stop token
            if tok_id == eos_token_id:
                break

            # 2. Yield the decoded token
            yield self.tokenizer.decode([tok_id])

            # 3. Generate the next token
            next_token, cache = step(next_token, cache)
            token_count += 1


if __name__ == "__main__":
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2")
    hf_sd = hf_model.state_dict()

    config = GPTModelConfig()

    mlx_model = GPT2(config)
    mlx_model.load_weights_from_hf_state_dict(hf_sd)
    del hf_sd, hf_model 

    prompt = "The history of machine learning begins"
    print(prompt, end=" >>> ")
    for text in mlx_model.predict(prompt, max_tokens=500):
        print(text, end="")
