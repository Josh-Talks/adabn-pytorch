import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import random
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.models.resnet import resnet18
from torchvision.models.vgg import vgg16_bn, vgg11_bn
from functools import partial

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Just trying to seed everything so I don't find myself looking confused at the screen
def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# Fake dataset class. Trying to be as fake as it can be
class ImageGeneratorDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.vector_dim = (3, 128, 128)
        self.data = []
        self.create_data()

    def create_data(self):
        for i in range(self.num_samples):
            # FIXED: Create some variation instead of all zeros
            self.data.append(torch.randn(self.vector_dim) * 0.1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]


# Simple model to understand the behavior of AdaBN passing through two BN layers
class BasicModel(nn.Module):

    def __init__(
        self,
    ):
        super(BasicModel, self).__init__()
        self.layer1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.layer2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=7, stride=2, padding=3
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)

        return x


# The hook class is responsible to store the BN outputs when the test dataloader is passed
class BatchNormStatHook(object):
    """
    Hook to accumulate statistics from BatchNorm layers during inference.
    """

    def __init__(self):
        self.bn_stats = {}  # Dictionary to store layer name and accumulated statistics

    def __call__(self, module, input, output, name):
        """
        Hook function called during the forward pass of BatchNorm layers.

        Args:
            module (nn.Module): The BatchNorm layer.
            input (torch.Tensor): Input tensor to the layer.
            output (torch.Tensor): Output tensor from the layer.
        """
        layer_name = name
        # Check if layer name already exists (multiple BN layers with same type)
        # But I think this might not be required if the model is well defined properly?
        # Not taking care of nn.Sequential

        if layer_name not in self.bn_stats:
            # FIXED: Initialize with proper structure for accumulation
            self.bn_stats[layer_name] = {"mean": 0, "var": 0, "count": 0}

        # CRITICAL FIX #1: Process INPUT tensor, not output!
        # Ensure output is not a view (avoid potential errors)
        x = input[0].clone().detach()  # FIXED: Use input[0] instead of output

        # Calculate mean and variance of the INPUT (not output)
        if x.dim() == 4:  # 2D: [N, C, H, W]
            dims = [0, 2, 3]
        elif x.dim() == 5:  # 3D: [N, C, D, H, W]
            dims = [0, 2, 3, 4]
        else:
            raise ValueError(
                f"Unsupported input dimension {x.dim()} for BatchNorm layer"
            )
        mean = x.mean(dims)  # FIXED: Process input activations
        var = x.var(dims, unbiased=False)

        # CRITICAL FIX #2: Do NOT sum across channels! Keep per-channel info
        batch_size = x.size(0)

        # Initialize accumulators with correct shape on first call
        if isinstance(self.bn_stats[layer_name]["mean"], int):
            self.bn_stats[layer_name]["mean"] = torch.zeros_like(mean)
            self.bn_stats[layer_name]["var"] = torch.zeros_like(var)

        # Update accumulated statistics for this layer (keep per-channel)
        self.bn_stats[layer_name]["mean"] += mean * batch_size  # FIXED: No .sum()!
        self.bn_stats[layer_name]["var"] += var * batch_size  # FIXED: No .sum()!

        # This might not be required, but still saving just in-case
        self.bn_stats[layer_name][
            "count"
        ] += batch_size  # FIXED: Count samples, not channels


def compute_bn_stats(model, dataloader):
    """
    Computes mean and variance of BatchNorm layer outputs across all images in the dataloader.

    Args:
      model (nn.Module): The trained model.
      dataloader (torch.utils.data.DataLoader): The dataloader for the data.

    Returns:
      dict: Dictionary containing layer names and their mean and variance statistics.
    """

    # CRITICAL FIX #3: Set model to TRAIN mode to ensure proper statistics computation!
    original_mode = model.training
    model.train()  # CORRECTED: Must be train() for proper AdaBN statistics

    # CRITICAL FIX #4: Temporarily disable dropout for consistent statistics
    # Store original dropout states and set them to eval mode
    dropout_modules = []
    original_dropout_modes = []
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_modules.append(module)
            original_dropout_modes.append(module.training)
            module.eval()  # Disable dropout during statistics computation

    # Create a hook instance
    hook = BatchNormStatHook()
    hook_handles = []

    # Register the hook on all BatchNorm layers in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            handle = module.register_forward_hook(partial(hook, name=name))
            hook_handles.append(handle)  # FIXED: Store handles for cleanup

    try:
        device = next(model.parameters()).device
        # Iterate through the dataloader
        with torch.no_grad():
            for data, _ in dataloader:
                # Forward pass (hook will accumulate statistics)
                model(data.to(device))

        # Calculate mean and variance for each layer
        final_stats = {}
        for layer_name, stats in hook.bn_stats.items():
            # print("Found the layer!!!")
            if stats["count"] > 0:
                mean = (
                    stats["mean"] / stats["count"]
                )  # FIXED: Now divides tensors properly
                var = stats["var"] / stats["count"]
                final_stats[layer_name] = {"mean": mean, "var": var}

    finally:
        # FIXED: Clean up hooks to prevent memory leaks
        for handle in hook_handles:
            handle.remove()
        model.train(original_mode)  # Restore original mode

        # CRITICAL FIX #4: Restore original dropout states
        for module, original_mode in zip(dropout_modules, original_dropout_modes):
            module.train(original_mode)

    # Return the accumulated statistics
    return final_stats


# Now replace the current stats with the computed one
def replace_bn_stats(model, bn_stats):
    with torch.no_grad():
        for name, module in model.named_modules():
            if name in bn_stats and isinstance(module, nn.BatchNorm2d):
                # FIXED: Add shape verification
                expected_shape = module.running_mean.shape
                computed_mean = bn_stats[name]["mean"]
                computed_var = bn_stats[name]["var"]

                if computed_mean.shape != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected {expected_shape}, got {computed_mean.shape}"
                    )

                print("Before---------------------------------------")
                print(module.running_mean)
                module.running_mean.data.copy_(
                    computed_mean.to(module.running_mean.device)
                )  # FIXED: Handle device
                module.running_var.data.copy_(
                    computed_var.to(module.running_var.device)
                )
                print(module.running_mean)
                print("After---------------------------------------")


class SequentialBatchNormStatHook(object):
    """
    Hook to accumulate statistics from a single BatchNorm layer during inference.
    Used for sequential BatchNorm adaptation.
    """

    def __init__(self, target_layer_name):
        self.target_layer_name = target_layer_name
        self.bn_stats = {"mean": 0, "var": 0, "count": 0}

    def __call__(self, module, input, output, name):
        """
        Hook function called during the forward pass of BatchNorm layers.
        Only accumulates statistics for the target layer.

        Args:
            module (nn.Module): The BatchNorm layer.
            input (torch.Tensor): Input tensor to the layer.
            output (torch.Tensor): Output tensor from the layer.
            name (str): Name of the layer.
        """
        # Only process the target layer
        if name != self.target_layer_name:
            return

        # Process INPUT tensor to get statistics
        x = input[0].clone().detach()

        # Calculate mean and variance of the INPUT
        if x.dim() == 4:  # 2D: [N, C, H, W]
            dims = [0, 2, 3]
        elif x.dim() == 5:  # 3D: [N, C, D, H, W]
            dims = [0, 2, 3, 4]
        else:
            raise ValueError(
                f"Unsupported input dimension {x.dim()} for BatchNorm layer"
            )

        mean = x.mean(dims)
        var = x.var(dims, unbiased=False)
        batch_size = x.size(0)

        # Initialize accumulators with correct shape on first call
        if isinstance(self.bn_stats["mean"], int):
            self.bn_stats["mean"] = torch.zeros_like(mean)
            self.bn_stats["var"] = torch.zeros_like(var)

        # Update accumulated statistics for this layer
        self.bn_stats["mean"] += mean * batch_size
        self.bn_stats["var"] += var * batch_size
        self.bn_stats["count"] += batch_size


def get_bn_layers_ordered(model):
    """
    Get all BatchNorm layers in the model in the order they appear during forward pass.

    Args:
        model (nn.Module): The model to analyze.

    Returns:
        list: List of tuples (layer_name, module) for all BatchNorm layers in order.
    """
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            bn_layers.append((name, module))
    return bn_layers


def compute_single_bn_stats(model, dataloader, target_layer_name):
    """
    Computes mean and variance for a single BatchNorm layer across all images in the dataloader.
    Uses a partial forward pass for efficiency when possible.

    IMPORTANT: This function sets the model to eval() mode so that previously updated
    BatchNorm layers use their updated running statistics, allowing sequential adaptation
    where each layer sees the effects of all previously adapted layers.

    Args:
        model (nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader for the data.
        target_layer_name (str): Name of the target BatchNorm layer.

    Returns:
        dict: Dictionary containing mean and variance statistics for the target layer.
    """
    # Set model to EVAL mode so that:
    # 1. Previously updated BatchNorm layers use their updated running statistics
    # 2. Dropout is automatically disabled for consistent statistics
    # 3. We compute statistics based on the current state of all previous layers
    # 4. This enables true sequential adaptation where each layer sees properly adapted inputs
    original_mode = model.training
    model.eval()

    # Create a hook instance for the target layer only
    hook = SequentialBatchNormStatHook(target_layer_name)
    hook_handle = None

    # Register the hook only on the target BatchNorm layer
    for name, module in model.named_modules():
        if name == target_layer_name and isinstance(module, torch.nn.BatchNorm2d):
            hook_handle = module.register_forward_hook(partial(hook, name=name))
            break

    if hook_handle is None:
        raise ValueError(
            f"Target layer '{target_layer_name}' not found or is not a BatchNorm2d layer"
        )

    try:
        device = next(model.parameters()).device

        # Create partial model for efficiency (stops computation at target layer)
        # partial_model = create_partial_forward_model(model, target_layer_name)

        # Iterate through the dataloader
        with torch.no_grad():
            for batch_data in dataloader:
                if isinstance(batch_data, (list, tuple)):
                    data = batch_data[0]  # Assume first element is input
                else:
                    data = batch_data
                # Forward pass up to target layer (hook will accumulate statistics)
                # partial_model(data.to(device))
                model(data.to(device))

        # Calculate final mean and variance for the target layer
        final_stats = {}
        if hook.bn_stats["count"] > 0:
            mean = hook.bn_stats["mean"] / hook.bn_stats["count"]
            var = hook.bn_stats["var"] / hook.bn_stats["count"]
            final_stats[target_layer_name] = {"mean": mean, "var": var}

    finally:
        # Clean up hooks
        if hook_handle:
            hook_handle.remove()
        # Restore original model mode
        model.train(original_mode)

    return final_stats


def update_single_bn_layer(model, layer_name, bn_stats):
    """
    Update statistics for a single BatchNorm layer.

    Args:
        model (nn.Module): The model containing the BatchNorm layer.
        layer_name (str): Name of the BatchNorm layer to update.
        bn_stats (dict): Dictionary containing mean and variance statistics.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if name == layer_name and isinstance(module, nn.BatchNorm2d):
                if layer_name in bn_stats:
                    # Verify shape compatibility
                    expected_shape = module.running_mean.shape
                    computed_mean = bn_stats[layer_name]["mean"]
                    computed_var = bn_stats[layer_name]["var"]

                    if computed_mean.shape != expected_shape:
                        raise ValueError(
                            f"Shape mismatch for {layer_name}: expected {expected_shape}, got {computed_mean.shape}"
                        )

                    # Update the layer's statistics
                    module.running_mean.data.copy_(
                        computed_mean.to(module.running_mean.device)
                    )
                    module.running_var.data.copy_(
                        computed_var.to(module.running_var.device)
                    )

                    print(f"Updated BatchNorm layer '{layer_name}' with new statistics")
                break


def sequential_bn_adaptation(model, dataloader, verbose=True):
    """
    Sequentially update BatchNorm statistics layer by layer.
    Each layer is updated based on the outputs of previously updated layers.

    Args:
        model (nn.Module): The trained model.
        dataloader (torch.utils.data.DataLoader): The dataloader for target domain data.
        verbose (bool): Whether to print progress information.

    Returns:
        dict: Dictionary containing final statistics for all updated layers.
    """
    # Get all BatchNorm layers in order
    bn_layers = get_bn_layers_ordered(model)

    if not bn_layers:
        if verbose:
            print("No BatchNorm layers found in the model")
        return {}

    if verbose:
        print(
            f"Found {len(bn_layers)} BatchNorm layers: {[name for name, _ in bn_layers]}"
        )
        print("Starting sequential BatchNorm adaptation...")

    all_updated_stats = {}

    # Process each BatchNorm layer sequentially
    for i, (layer_name, layer_module) in enumerate(bn_layers):
        if verbose:
            print(f"\nStep {i+1}/{len(bn_layers)}: Processing layer '{layer_name}'")

        # Compute statistics for current layer using current model state
        layer_stats = compute_single_bn_stats(model, dataloader, layer_name)

        if layer_stats:
            # Update the current layer with new statistics
            update_single_bn_layer(model, layer_name, layer_stats)
            all_updated_stats.update(layer_stats)

            if verbose:
                mean_val = layer_stats[layer_name]["mean"]
                var_val = layer_stats[layer_name]["var"]
                print(
                    f"  Updated '{layer_name}' - Mean range: [{mean_val.min():.4f}, {mean_val.max():.4f}]"
                )
                print(
                    f"  Updated '{layer_name}' - Var range: [{var_val.min():.4f}, {var_val.max():.4f}]"
                )
        else:
            if verbose:
                print(f"  Warning: No statistics computed for layer '{layer_name}'")

    if verbose:
        print(
            f"\nSequential BatchNorm adaptation completed. Updated {len(all_updated_stats)} layers."
        )

    return all_updated_stats
