import torch
import torch.nn as nn
from typing import List, Tuple

# add source to this code with reasoning for doing all this shit


class LayerwiseInitializer:
    """Handles initialization of parameters with different scales per layer"""

    @staticmethod
    def get_scale(name: str) -> float:
        """Determine appropriate scale based on parameter name."""
        if "bias" in name or any(x in name for x in ["output", "action", "out"]):
            return 0.01
        return 1.0  # default scale for hidden layers

    @staticmethod
    def initialize_flat_params(model: nn.Module) -> torch.Tensor:
        """
        Initialize parameters as a flat vector with appropriate scales per layer.
        """
        # Gather all parameters
        params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in params)
        flat_params = torch.zeros(total_params, device=next(model.parameters()).device)

        # Initialize each section with appropriate scale
        offset = 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_shape = param.numel()
            scale = LayerwiseInitializer.get_scale(name)

            # Initialize with scaled normal distribution
            init_params = torch.randn(param_shape, device=flat_params.device)
            init_params *= scale / (torch.norm(init_params) + 1e-8)

            # Store in flat vector
            flat_params[offset : offset + param_shape] = init_params
            offset += param_shape

        return flat_params

    @staticmethod
    def print_layer_info(model: nn.Module):
        """Print initialization info for debugging"""
        print("\nLayer initialization scales:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                scale = LayerwiseInitializer.get_scale(name)
                print(f"{name}: scale={scale}")
