# This is a modified version of the GPT2 model from the Hugging Face Transformers library.
# For the original code, see:
# https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/models/gpt2/modeling_gpt2.py

from typing import Any, Callable, Dict, Optional, Tuple, Union, override

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2Model
from transformers.modeling_outputs import ModelOutput
from transformers.models.gpt2.modeling_gpt2 import (
    ALL_ATTENTION_FUNCTIONS,
    GPT2MLP,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    GPT2Attention,
    GPT2Block,
    GPT2Config,
    GPT2LMHeadModel,
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
    eager_attention_forward,
    logging,
)

from utils.flash_attention_helper import compilable_flash_attention_2_forward

logger = logging.get_logger(__name__)
ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = compilable_flash_attention_2_forward # pyright: ignore[reportArgumentType]

class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) as introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding" by Su et al.
    https://arxiv.org/abs/2104.09864.
    """
    def __init__(self, dim: int, max_position_embeddings=2048, base=10000):
        """
        Rotary Positional Embedding (RoPE). The class computes the cosine and sine embeddings for the positional encodings and caches them.
        Args:
            dim (int): The dimension of the input tensor.
            max_position_embeddings (int): The maximum position embeddings to cache.
            base (int): The base of the sinusoidal function.
            device (torch.device): The device to use.
        """
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.inv_freq: torch.FloatTensor
        self._initialize(max_position_embeddings)

    def _initialize(self, max_seq_len: int):
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos())
        self.cos_cached: torch.FloatTensor
        self.register_buffer("sin_cached", freqs.sin())
        self.sin_cached: torch.FloatTensor

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): The input tensor.
            seq_len (int): The sequence length.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The cosine and sine embeddings.
        """
        return (
            self.cos_cached[position_ids, ...].unsqueeze(1),
            self.sin_cached[position_ids, ...].unsqueeze(1),
        )

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate the tensor by 90 degrees.
    Args:
        x (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The rotated tensor.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotary positional embeddings to the tensor.
    Args:
        x (torch.Tensor): The input tensor.
        cos (torch.Tensor): The cosine embeddings.
        sin (torch.Tensor): The sine embeddings.
    Returns:
        torch.Tensor: The tensor with the positional embeddings applied.
    """
    x1, x2 = x.chunk(2, dim=-1)
    # Apply rotation: [cos -sin; sin cos] @ [x1; x2]
    
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class RopeGPT2Attention(GPT2Attention):
    @override
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)

        query_states = query_states.view(shape_q).transpose(1, 2)
        key_states = key_states.view(shape_kv).transpose(1, 2)
        value_states = value_states.view(shape_kv).transpose(1, 2)

        ##############################################
        # QK-norm
        norm_func = torch.nn.functional.rms_norm if self.config.norm_class == "rms" else torch.nn.functional.layer_norm
        query_states = norm_func(query_states, (query_states.size(-1),)).to(query_states.dtype)
        key_states = norm_func(key_states, (key_states.size(-1),)).to(query_states.dtype)
        ##############################################

        ##############################################
        # RoPE
        if cos is not None and sin is not None:
            query_states = apply_rotary_pos_emb(query_states, cos, sin)
            key_states = apply_rotary_pos_emb(key_states, cos, sin)
        ##############################################

        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key, key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)

        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None

        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                # Attention functions are consistent with previous equivalent attention classes, however they do not support some options
                # (e.g. layer scaling, head mask) that eager supports. These implementations are thus equivalent to previous code, but
                # not necessarily to eager (if mentionned options are provided).
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    
class RopeGPT2Block(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx=layer_idx)
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        
        norm_class = nn.RMSNorm if config.norm_class=="rms" else nn.LayerNorm
        self.ln_1 = norm_class(hidden_size)
        self.attn: RopeGPT2Attention = RopeGPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = norm_class(hidden_size)

        if config.add_cross_attention:
            self.crossattention: RopeGPT2Attention = RopeGPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = norm_class(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = GPT2MLP(inner_dim, config)

    @override
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cos: Optional[torch.FloatTensor]=None,
        sin: Optional[torch.FloatTensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            sin=sin,
            cos=cos,
            position_ids=position_ids,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                sin=sin,
                cos=cos,
                position_ids=position_ids,
                cu_seq_lens=cu_seq_lens,
                max_seq_length=max_seq_length,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class RopeGPT2Model(GPT2Model):
    """
    A GPT2 model that uses the Rotary Positional Embedding (RoPE) instead of the standard sinusoidal positional embeddings.
    """
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        del self.h
        self.h = nn.ModuleList([RopeGPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.rotary_emb = RotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings
        )
        del self.wpe
        self.wpe = lambda x: 0  # Dummy positional embeddings
        norm_class = nn.RMSNorm if config.norm_class=="rms" else nn.LayerNorm
        self.ln_f = norm_class(config.hidden_size)
        self.post_init()

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # pyright: ignore[reportRedeclaration]
        attention_mask: Optional[torch.FloatTensor] = None, # pyright: ignore[reportRedeclaration]
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None, # pyright: ignore[reportRedeclaration]
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            device = input_ids.device
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0] # pyright: ignore[reportOptionalMemberAccess]
        elif inputs_embeds is not None:
            device = inputs_embeds.device
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values: Tuple[Tuple[torch.Tensor]] | Tuple[Tuple[None]] = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0) # pyright: ignore[reportOptionalMemberAccess]

        if inputs_embeds is None:
            inputs_embeds: torch.FloatTensor = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        ######################################
        # RoPE
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        ######################################

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            # attention_mask = attention_mask if (attention_mask is not None and not attention_mask.bool().all()) else None
            pass # avoid compile graph break
        elif _use_sdpa:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask=attention_mask,
                input_shape=(batch_size, input_shape[-1]),
                inputs_embeds=inputs_embeds,
                past_key_values_length=past_length,
            )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask: torch.FloatTensor = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask: torch.FloatTensor = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask: torch.FloatTensor = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1] # pyright: ignore[reportArgumentType]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask) # pyright: ignore[reportArgumentType]
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i in range(len(self.h)):
            block, layer_past = self.h[i], past_key_values[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past if past_state is not None)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,) # pyright: ignore[reportOptionalOperand]

            if self.gradient_checkpointing and self.training:
                ######################################
                # RoPE
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i], # pyright: ignore[reportOptionalSubscript]
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                    sin=sin,
                    cos=cos,
                    position_ids = position_ids
                )
                ######################################
            else:
                ######################################
                # RoPE
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    sin=sin,
                    cos=cos,
                    position_ids = position_ids,
                    cu_seq_lens=cu_seq_lens,
                    max_seq_length=max_seq_length,
                )
                ######################################

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],) # pyright: ignore[reportOptionalOperand]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],) # pyright: ignore[reportOptionalOperand]
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],) # pyright: ignore[reportOptionalOperand]

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items(): # pyright: ignore[reportOptionalMemberAccess]
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,) # pyright: ignore[reportOptionalOperand]

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents, # pyright: ignore[reportArgumentType]
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class RopeGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.transformer = RopeGPT2Model(config)
        self.post_init()
        self.loss_fct = CrossEntropyLoss(reduction="none")

    @override
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None, position_ids=None, cu_seq_lens=None, max_seq_length=None, **kwargs) -> dict[str, Any]:
        # This is necessary to supress a Value error because of unused generation arguments.
        return {
            **super().prepare_inputs_for_generation(
                input_ids,
                past_key_values=None, # This causes trouble when past_key_values is not None
                inputs_embeds=inputs_embeds,
                **kwargs),
            "attention_mask": attention_mask,
            "position_ids": position_ids, # GPT ereases the position_ids from the kwargs for some reason
            "cu_seq_lens": cu_seq_lens,
            "max_seq_length": max_seq_length
        }

    @override
    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> Dict[str, Any]:
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            num_new_tokens=num_new_tokens
        )
        attention_mask: torch.LongTensor =model_kwargs["attention_mask"]
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        model_kwargs["position_ids"] = position_ids
        model_kwargs["max_seq_length"] = position_ids.max().item() + 1
        return model_kwargs

    @override
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seq_lens: Optional[torch.Tensor] = None,
        max_seq_length: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seq_lens=cu_seq_lens,
            max_seq_length=max_seq_length,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous() # pyright: ignore[reportOptionalSubscript]
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        out = CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        return out
