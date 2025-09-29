if __name__ == "__main__":
    # Add project root to path when running directly 
    import sys
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from math import sqrt
from typing import Literal, Optional, override

import torch
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from networks.number_embedding_modules.abc_embedding import ABCEmbedding
from utils.dynamic_compile import dynamic_compile


class FourierFloat(ABCEmbedding):
    def __init__(self, n_embed: int, scaling: Literal["log", "linear"]="log", float_type: Optional[Literal["float64", "float32"]]=None, base=2, norm: Literal["rms_norm", "layer_norm"]="rms_norm", loss="mse", device="cuda:0"):
        super().__init__()
        self.base = base
        if scaling == "log":
            if float_type == "float64":
                self.exponent_bits = 14
                self.mantissa_bits = 52
                
                finfo = torch.finfo(torch.float64)
            elif float_type == "float32":
                self.exponent_bits = 11
                self.mantissa_bits = 23
                finfo = torch.finfo(torch.float32)
            else:
                ValueError(f"Unknown float type: {float_type}. Valid options are 'float64' and 'float32'.")
            
            self.MAX, self.TINY = finfo.max*(1-1e-11), finfo.tiny*2.
            self.single_shift: int = self.base**(self.exponent_bits-4.)
            self.double_shift: int = self.base**(self.exponent_bits-3.)
            self.xi = self._xi_log
            self.xi_inv = self._inv_xi_log
        elif scaling == "linear":
            self.exponent_bits = 52
            self.mantissa_bits = 52
            self.MAX, self.TINY = 1e15, 1e-14
            self.single_shift: int = 0
            self.double_shift: int = self.base**self.exponent_bits
            self.xi = self._xi_id
            self.xi_inv = self._inv_xi_id
        else:
            raise ValueError(f"Unknown scaling type: {scaling}. Valid options are 'log' and 'linear'.")
            
        self.phi = torch.linspace(-self.mantissa_bits, self.exponent_bits-1, self.exponent_bits+self.mantissa_bits, dtype=torch.float64, device=device).unsqueeze(0)
        self.wave_lengths = (self.base ** self.phi)
        self.half_wave_lengths = self.wave_lengths / 2.
        self.quarter_wave_lengths = self.wave_lengths / 4.
        self.frequencies = (2.*torch.pi) / self.wave_lengths
        self.shift_mod_wave_lengths: torch.DoubleTensor = self.single_shift % self.wave_lengths # We shift number<1 with negative exponents to the right to make room for negative number. To avoid precision loss, we only shift the largest frequencies
        
        self.norm = norm
        self.scale_factor = sqrt(2) if norm == "layer_norm" else 1
        self.freq_size = self.phi.shape[1]
        self.pad_size= n_embed - (2*self.freq_size) * (2 if norm == "layer_norm" else 1)
        self.num_head = torch.compile(torch.nn.Sequential(
            torch.nn.Linear(n_embed, self.freq_size*2),
            torch.nn.Tanh()
        ), dynamic=True)
        if loss == "l1":
            self.loss_func = torch.nn.L1Loss(reduction="none")
        elif loss == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unknown loss function: {loss}")
        
        self.freq_loss_weights: torch.FloatTensor | torch.BFloat16Tensor = torch.ones((1, self.freq_size), device=device, dtype=torch.float32).tile(2)
        self.device = device

    def _xi_log(self, x: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Computes the logarithm of the absolute value of x and returns the exponent and mantissa.
        Args:
            x (torch.DoubleTensor): A tensor of shape (B,) containing the input numbers.
        Returns:
            exponent (torch.DoubleTensor): A tensor of shape (B, 1) containing the exponent of the logarithm.
            mantissa (torch.DoubleTensor): A tensor of shape (B, 1) containing the mantissa of the logarithm.
        """
        exponent = torch.where(
            x.abs()>=self.TINY,
            torch.log2(x.abs()).trunc(),
            torch.full_like(x, -self.single_shift)
        )
        mantissa = torch.where(
            x.abs()>=self.TINY,
            torch.sign(x)*(torch.log2(x.abs() * 2**exponent.neg())),
            torch.zeros_like(x)
        )
        return exponent.unsqueeze(-1), mantissa.unsqueeze(-1)

    def _inv_xi_log(self, exponent: torch.DoubleTensor, mantissa: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Inverse xi function for the given exponent and mantissa.
        """
        sign = torch.sign(mantissa+exponent)
        recon = sign * ((2. ** (sign*exponent - self.single_shift)) * (2 ** (sign*mantissa)))
        # Set all values smaller than TINY to zero
        recon = torch.where(
            recon.abs() < self.TINY,
            torch.zeros_like(recon),
            recon
        )
        return recon
    
    def _xi_id(self, x: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Computes the identity function for the given input x and returns the int value as the exponent and fraction value as the mantissa.
        Args:
            x (torch.DoubleTensor): A tensor of shape (B,) containing the input numbers.
        Returns:
            exponent (torch.DoubleTensor): A tensor of shape (B, 1) containing the exponent of the identity function.
            mantissa (torch.DoubleTensor): A tensor of shape (B, 1) containing the mantissa of the identity function.
        """
        exponent = torch.where(
            x.abs()>=self.TINY,
            x.trunc(),
            torch.zeros_like(x)
        )
        mantissa = torch.where(
            x.abs()>=self.TINY,
            x - exponent,
            torch.zeros_like(x)
        )
        return exponent.abs().unsqueeze(-1), mantissa.unsqueeze(-1)

    def _inv_xi_id(self, exponent: torch.DoubleTensor, mantissa: torch.DoubleTensor) -> torch.DoubleTensor:
        """
        Inverse of the xi_id function for the given exponent and mantissa.
        Args:
            exponent (torch.DoubleTensor): A tensor of shape (B, 1) containing the exponent.
            mantissa (torch.DoubleTensor): A tensor of shape (B, 1) containing the mantissa.
        Returns:
            torch.DoubleTensor: A tensor of shape (B,) containing the reconstructed values.
        """
        recon = mantissa + exponent
        recon = torch.where(
            recon.abs() < self.TINY,
            torch.zeros_like(recon),
            recon
        )
        return recon

    @override
    @torch.compile(dynamic=True)
    def forward(self, x: torch.DoubleTensor) -> torch.FloatTensor | torch.BFloat16Tensor:
        assert not (0 < x.abs().min() < self.TINY ).any(), f"Input values are too small. Smallest given value: {x.abs().min()}, allowed: {self.TINY}"
        assert (x.abs().max() <= self.MAX).all(), f"Input values are too large. Largest given value: {x.abs().max()}, allowed: {self.MAX}"
        # To avoid precision loss caused by exponents > 1 or exponents < -1, we encode the exponent and mantissa separately
        exponent, mantissa = self.xi(x)
        exp_mod_wave_length = torch.sign(x).unsqueeze(-1) * ((exponent % self.wave_lengths)+ self.shift_mod_wave_lengths)
        exp_enc = mantissa * self.frequencies + exp_mod_wave_length * self.frequencies
        enc = torch.cat([torch.sin(exp_enc), torch.cos(exp_enc)], dim=-1).to(dtype=torch.get_autocast_dtype("cuda"))
        return enc

    @override
    @dynamic_compile()
    def combine_embeds(
            self,
            inputs_embeds: torch.FloatTensor | torch.BFloat16Tensor,
            num_encoding: torch.FloatTensor | torch.BFloat16Tensor,
            number_mask: torch.BoolTensor,
        ) -> torch.FloatTensor | torch.BFloat16Tensor:
        if self.norm == "rms_norm":
            num_embedding = num_encoding
        elif self.norm == "layer_norm":
            # This is necessary to ensure zero mean and unit variance for the number encoding
            num_embedding = self.scale_factor * torch.concat((num_encoding, -num_encoding), dim=-1)
        else:
            raise ValueError(f"Unknown norm type: {self.norm}")
        combined_embeds = inputs_embeds + torch.nn.functional.pad(num_embedding, (0, self.pad_size))
        return combined_embeds

    @override
    def compute_num_loss(
        self,
        out: CausalLMOutputWithCrossAttentions,
        num_encodings: torch.FloatTensor | torch.BFloat16Tensor,
        number_mask: torch.BoolTensor,
        numbers: torch.DoubleTensor,
        hidden_states_slice=slice(0,-1),
        **kwargs
    ) -> torch.FloatTensor:
        assert out.hidden_states is not None, "Model output does not contain hidden states. Please set output_hidden_states=True in the model configuration."
        num_preds: torch.FloatTensor = self.num_head(out.hidden_states[-1][:, hidden_states_slice][number_mask])
        loss: torch.FloatTensor = self.loss_func(num_preds.float(), num_encodings[number_mask][:,:2*self.freq_size].float())
        num_loss_per_frequency = loss.detach().view(-1, self.freq_size).mean(0)
        num_loss_per_sample = torch.zeros_like(number_mask, dtype=self.freq_loss_weights.dtype)
        # Weight the absolute error per frequency by their logarithmic value
        num_loss_per_sample[number_mask] = (self.freq_loss_weights * loss).mean(-1) / self.freq_loss_weights.mean()
        train_metrics = {
            "num_loss": num_loss_per_sample,
            "num_loss_per_frequency": num_loss_per_frequency,
            "loss_log": loss.detach().mean().item()
        }
        return train_metrics
    
    @override
    def decode(self, out: CausalLMOutputWithCrossAttentions, number_mask: torch.BoolTensor, scale_factor: float=9) -> torch.DoubleTensor:
        assert out.hidden_states is not None, "Model output does not contain hidden states. Please set output_hidden_states=True in the model configuration."
        encoding: torch.BFloat16Tensor = self.num_head(out.hidden_states[-1][...,-1:,:][number_mask])
        sin = encoding[:, :self.freq_size].double()  # Shape: (B, num_fourier)
        cos = encoding[:, self.freq_size:self.freq_size*2].double()   # Shape: (B, num_fourier)
        
        # Calculate asin and acos for stability
        asin = torch.asin(sin) / torch.pi
        acos = torch.acos(cos) / torch.pi
        
        # Determine whether to use sine or cosine for each frequency
        use_sin = torch.sigmoid(scale_factor * (torch.abs(asin) - torch.abs(acos - 0.5)))

        # Precompute phase values
        acos_freq = acos * self.half_wave_lengths
        acos_freq_inv = (1. - acos) * self.half_wave_lengths
        asin_freq = asin * self.half_wave_lengths
        asin_freq_inv = - asin * self.half_wave_lengths
        estimate: torch.DoubleTensor = (asin_freq[:, -1] + (acos_freq[:, -1] * asin_freq[:, -1].sign()))/2
        exponent: torch.DoubleTensor = torch.zeros_like(estimate)
        # Iterate over the Fourier frequencies in reverse
        for j in range(self.freq_size - 2, -1, -1):
            if j==self.mantissa_bits-1: # Once the exponent is determined, we subtract it to only decode the mantissa
                estimate = estimate.sign()*(estimate.abs() % self.double_shift)
                exponent = estimate.trunc().double()
                estimate=estimate-exponent # Now represents the mantissa
                
            translated = estimate + self.quarter_wave_lengths[:,j]
            sin_falling = torch.div(translated, self.half_wave_lengths[:,j], rounding_mode="floor") % 2.
            sin_rising = 1. - sin_falling
            sin_mod = torch.remainder(translated, self.half_wave_lengths[:,j])
            cos_falling = torch.div(estimate, self.half_wave_lengths[:,j], rounding_mode="floor") % 2.
            cos_rising = 1. - cos_falling
            cos_mod = torch.remainder(estimate, self.half_wave_lengths[:,j])
            # If acos/asin wave is decreasing (base % 2 == 1), we take the invert phase
            change = (
                    use_sin[:,j]     * (asin_freq[:, j] * sin_rising + asin_freq_inv[:, j] * sin_falling - sin_mod + self.quarter_wave_lengths[:,j])
                + (1 - use_sin[:,j]) * (acos_freq[:, j] * cos_rising + acos_freq_inv[:, j] * cos_falling - cos_mod)
            )
            estimate = estimate + change

        recon: torch.DoubleTensor = self.xi_inv(exponent=exponent, mantissa=estimate)
        assert not recon.isinf().any(), "Encountered inf values during decoding. This is likely due to an overflow in the exponent. Please check the input values and the exponent bits."
        assert not recon.isnan().any(), "Encountered NaN values during decoding. This is likely due to an overflow in the exponent. Please check the input values and the exponent bits."
        return recon

if __name__ == "__main__":
    SCALING: Literal["log", "linear"] = "log"               # TODO choose scaling
    DTYPE=torch.float64                                     # TODO choose dtype
    BASE: int = 2                                           # TODO choose base for frequencies

    BOTTLENECK_DTYPE = torch.bfloat16                       # TODO choose dtype for bottleneck
    # BOTTLENECK_DTYPE = torch.float8_e4m3fn

    # Test the FourierFloat class using the reusable test utility
    from test_utils import run_dtype_test
    
    # Test parameters
    BOTTLENECK_DTYPE = torch.bfloat16
    
    # Initialize embedding
    float_type: Literal['float64', 'float32'] = str(DTYPE).removeprefix("torch.")
    fourier_float = FourierFloat(n_embed=384, scaling=SCALING, base=BASE, float_type=float_type, device="cpu").to(dtype=BOTTLENECK_DTYPE)
    
    results = run_dtype_test(embedding_module=fourier_float, dtype=torch.float64)