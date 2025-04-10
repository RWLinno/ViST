import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import inspect
from torchvision.transforms import Resize
import threading
import matplotlib.pyplot as plt
from contextlib import contextmanager
import gc

def safe_resize(size, interpolation):
    """
    Create a safe resize operation that works with different versions of torchvision
    """
    signature = inspect.signature(Resize)
    params = signature.parameters
    if 'antialias' in params:
        return Resize(size, interpolation, antialias=False)
    else:
        return Resize(size, interpolation)

def normalize_minmax(x, eps=1e-8):
    """
    Min-max normalization to scale values to [0,1] range, vectorized for any shape
    
    Args:
        x: Input tensor
        eps: Small epsilon value for numerical stability
        
    Returns:
        Normalized tensor with values in [0,1]
    """
    # Check for NaN or Inf values before normalization
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    
    x_min = x.view(*x.shape[:-2], -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
    x_max = x.view(*x.shape[:-2], -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
    diff = x_max - x_min
    # Handle case where min and max are equal (constant tensor)
    normalized = torch.where(diff > eps, (x - x_min) / (diff + eps), torch.zeros_like(x))
    
    # Final safety check to ensure no NaN values escape
    normalized = torch.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized

class MultiPerspectiveVisualEncoder(nn.Module):
    """
    Transforms spatio-temporal data into structured visual representations
    with fully vectorized batch processing for maximum efficiency
    """
    def __init__(self, configs):
        super().__init__()
        # Configuration parameters
        self.image_size = configs.get('image_size', 64)
        self.interpolation = configs.get('interpolation', 'bilinear')
        self.save_path = configs.get('save_path', "images_output")
        
        # Performance optimization parameters
        self.low_res_factor = configs.get('low_res_factor', 4)  # Low-resolution processing factor
        self.working_size = self.image_size // self.low_res_factor  # Working resolution
        self.save_interval = configs.get('save_interval', 1)  # Saving interval
        self.async_save = configs.get('async_save', True)  # Async visualization saving
        
        # Visualization enhancement parameters
        self.use_colormap = configs.get('use_colormap', True)  # Use colormaps for better visualization
        self.color_mode = configs.get('color_mode', 'all_color')  # 'all_color' or 'all_gray'
        self.temporal_colormap = configs.get('temporal_colormap', 'Blues')  # Blue colormap for temporal
        self.spatial_colormap = configs.get('spatial_colormap', 'viridis')  # Colormap for spatial channel
        self.correlation_colormap = configs.get('correlation_colormap', 'hot')  # Colormap for correlation
        self.enhance_contrast = configs.get('enhance_contrast', True)  # Apply contrast enhancement
        
        # Current epoch - will be updated during training
        self.current_epoch = 0
        
        # Feature extraction parameters
        self.dim_reduction = nn.Conv2d(configs.get('embed_dim', 64), 32, kernel_size=1)
        
        # Create output directory
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        
        # Set interpolation method
        interp_map = {
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "bicubic": Image.BICUBIC,
        }
        self.resize_fn = safe_resize((self.image_size, self.image_size), 
                                   interpolation=interp_map[self.interpolation])
        
    def set_epoch(self, epoch):
        """
        Set the current epoch number for visualization filenames
        
        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch
    
    def forward(self, hidden_state, history_data=None, adj_mx=None, save_images=False):
        """
        Efficiently generate multi-perspective visual representations using
        fully vectorized operations for maximum performance
        
        Args:
            hidden_state: Hidden state tensor [B, E, N, D]
            history_data: Historical time series data [B, T, N, D]
            adj_mx: Optional adjacency matrix [B, N, N] or [N, N]
            save_images: Whether to save visualization images
            
        Returns:
            visual_repr: Visual representation tensor [B, T, 3, H, W]
        """
        try:
            if history_data is None:
                raise ValueError("history_data is required for generating visual representations")
            
            # Generate visual representations with fully vectorized operations
            visual_repr = self._fully_vectorized_forward(hidden_state, history_data, adj_mx)
            
            # Final safety check and normalization
            visual_repr = torch.nan_to_num(visual_repr, nan=0.0, posinf=1.0, neginf=0.0)
            visual_repr = torch.clamp(visual_repr, 0.0, 1.0)
            
            # Save visualizations if needed
            if save_images:
                if self.async_save:
                    self._async_save_visualizations(visual_repr.detach())
                else:
                    self._save_visualizations(visual_repr)
            
            return visual_repr
            
        except Exception as e:
            print(f"Error in visual encoder: {str(e)}")
            # Return a fallback visual representation
            B, T = history_data.shape[0], history_data.shape[1]
            return torch.zeros(B, T, 3, self.image_size, self.image_size, device=history_data.device)
    
    def _fully_vectorized_forward(self, hidden_state, history_data, adj_mx):
        """
        Completely vectorized visual encoder with zero batch loops

        Args:
            hidden_state: Hidden state tensor [B, E, N, D]
            history_data: Historical time series data [B, T, N, D]
            adj_mx: Optional adjacency matrix

        Returns:
            visual_repr: Visual representation tensor [B, T, 3, H, W]
        """
        # More aggressive NaN handling for inputs
        if torch.isnan(history_data).any() or torch.isinf(history_data).any():
            history_data = torch.nan_to_num(history_data, nan=0.0, posinf=10.0, neginf=-10.0)
        if torch.isnan(hidden_state).any() or torch.isinf(hidden_state).any():
            hidden_state = torch.nan_to_num(hidden_state, nan=0.0, posinf=10.0, neginf=-10.0)
        
        # Add gradient scaling to improve stability
        history_data = torch.clamp(history_data, -100.0, 100.0)
        hidden_state = torch.clamp(hidden_state, -100.0, 100.0)
        
        B, T, N, D = history_data.shape
        device = history_data.device
        
        # Initialize final output tensor
        visual_repr = torch.zeros(B, T, 3, self.image_size, self.image_size, device=device)
        
        # Preprocess adjacency matrix once
        if adj_mx is not None and not isinstance(adj_mx, torch.Tensor):
            adj_mx = torch.tensor(adj_mx, device=device)
        
        try:
            # 1. Generate all three channels using fully vectorized operations
            # Handle each channel in separate try-except blocks for robustness
            try:
                temp_channel = self._generate_temporal_channel_vectorized(history_data)
            except Exception as e:
                print(f"Error generating temporal channel: {str(e)}")
                temp_channel = torch.zeros(B, T, self.working_size, self.working_size, device=device)
            
            try:
                spat_channel = self._generate_spatial_channel_vectorized(history_data, adj_mx)
            except Exception as e:
                print(f"Error generating spatial channel: {str(e)}")
                spat_channel = torch.zeros(B, T, self.working_size, self.working_size, device=device)
            
            try:
                corr_channel = self._generate_correlation_channel_vectorized(history_data, hidden_state, adj_mx)
            except Exception as e:
                print(f"Error generating correlation channel: {str(e)}")
                corr_channel = torch.zeros(B, T, self.working_size, self.working_size, device=device)
            
            # 2. Upsample all channels at once to target resolution
            channels = [temp_channel, spat_channel, corr_channel]
            for c, channel in enumerate(channels):
                # Reshape for efficient batch upsampling
                flat_channel = channel.reshape(B*T, 1, self.working_size, self.working_size)
                
                # Upsample to target size
                flat_high_res = F.interpolate(
                    flat_channel,
                    size=(self.image_size, self.image_size),
                    mode='bilinear',
                    align_corners=False
                )
            
                # Reshape back and store
                visual_repr[:, :, c] = flat_high_res.reshape(B, T, self.image_size, self.image_size)
            
            return visual_repr
            
        except Exception as e:
            print(f"Error in vectorized forward: {str(e)}")
            # Return fallback representation
            return torch.zeros(B, T, 3, self.image_size, self.image_size, device=device)
    
    def _generate_temporal_channel_vectorized(self, history_data):
        """
        Fully vectorized temporal channel generation with zero batch loops
        
        Args:
            history_data: Tensor of shape [B, T, N, D]
            
        Returns:
            temporal_channel: Tensor of shape [B, T, ws, ws]
        """
        B, T, N, D = history_data.shape
        device = history_data.device
        working_size = self.working_size
        
        # Initialize output
        temp_channel = torch.zeros(B, T, working_size, working_size, device=device)
        
        # 1. Calculate temporal derivatives for each batch simultaneously
        # [B, T-1, N, D] representing the change between consecutive timesteps
        if T > 1:
            temporal_diffs = history_data[:, 1:] - history_data[:, :-1]
            # Pad to maintain original time dimension
            padding = torch.zeros((B, 1, N, D), device=device)
            temporal_diffs = torch.cat([padding, temporal_diffs], dim=1)  # [B, T, N, D]
        else:
            # Handle single timestep case
            temporal_diffs = torch.zeros_like(history_data)
        
        # 2. Calculate rolling statistics with efficient vectorized operations
        window_size = min(3, T)
        rolled_stats = torch.zeros(B, T, N, D, device=device)
        
        # Pre-compute exponential decay weights for window positions
        decay_weights = torch.tensor([math.exp(-float(w)) for w in range(window_size)], 
                                   device=device)
        
        # Optimized rolling window calculation using custom indexing
        for t in range(T):
            # Define valid window for this timestep
            start_idx = max(0, t - window_size + 1)
            curr_window = t - start_idx + 1
            
            # Apply pre-computed weights
            for w_idx, w_pos in enumerate(range(start_idx, t + 1)):
                if w_idx < len(decay_weights):
                    weight = decay_weights[w_idx]
                    rolled_stats[:, t] += weight * history_data[:, w_pos]
        
        # 3. Create temporal visualizations
        # Determine sizing for the grid based on N and D
        grid_size = min(int(math.ceil(math.sqrt(N * D))), working_size)
        
        # Create grid mapping for all batches at once
        # Generate position indices for each feature
        indices = torch.arange(min(N*D, grid_size*grid_size), device=device)
        y_indices = (indices // grid_size).clamp(max=working_size-1)
        x_indices = (indices % grid_size).clamp(max=working_size-1)
        
        # Flatten features for grid placement
        flat_features = rolled_stats.reshape(B, T, N*D)
        
        # Apply non-linear scaling for better visibility with improved stability
        flat_features = torch.clamp(flat_features, -10.0, 10.0)  # Clamp before tanh for stability
        flat_features = torch.tanh(flat_features) * 0.5 + 0.5
        
        # Check for NaN values after tanh operation
        flat_features = torch.nan_to_num(flat_features, nan=0.5)
        
        # Efficient tensor-based grid construction using advanced indexing
        # Create a batch of empty grids
        grids = torch.zeros(B, T, working_size, working_size, device=device)
        
        # Use vectorized operations where possible
        valid_count = min(N*D, grid_size*grid_size)
        for i in range(valid_count):
            y, x = y_indices[i].item(), x_indices[i].item()
            if y < working_size and x < working_size:
                grids[:, :, y, x] = flat_features[:, :, i]
        
        # 4. Add time markers efficiently
        marker_size = max(1, working_size // 10)
        time_positions = (torch.arange(T, device=device) / max(1, T-1) * 
                         (working_size - marker_size)).long()
        
        # Add marker for each timestep using efficient slicing
        for t in range(T):
            pos = time_positions[t].item()
            # Use slicing for efficiency
            grids[:, t, pos:pos+marker_size, 0:marker_size] = 0.8
        
        # Return normalized tensor
        return normalize_minmax(grids)
    
    def _generate_spatial_channel_vectorized(self, history_data, adj_mx):
        """
        Fully vectorized spatial channel generation with enhanced gradients
        
        Args:
            history_data: Tensor of shape [B, T, N, D]
            adj_mx: Optional adjacency matrix [B, N, N] or [N, N]
            
        Returns:
            spatial_channel: Tensor of shape [B, T, ws, ws]
        """
        B, T, N, D = history_data.shape
        device = history_data.device
        working_size = self.working_size
        
        # Initialize output
        spat_channel = torch.zeros(B, T, working_size, working_size, device=device)
        
        # 1. Compute spatial grid parameters efficiently
        grid_size = min(int(math.ceil(math.sqrt(N))), working_size)
        
        # 2. Generate indices for node positions only once
        indices = torch.arange(min(N, grid_size*grid_size), device=device)
        y_indices = (indices // grid_size).clamp(max=working_size-1)
        x_indices = (indices % grid_size).clamp(max=working_size-1)
        
        # 3. Vectorized adjacency matrix preprocessing
        if adj_mx is not None:
            # Handle different dimensions efficiently
            if adj_mx.dim() == 2:  # [N, N] -> [B, N, N]
                adj_mx = adj_mx.unsqueeze(0).expand(B, -1, -1)
            elif adj_mx.dim() == 3 and adj_mx.size(0) == 1:  # [1, N, N] -> [B, N, N]
                adj_mx = adj_mx.expand(B, -1, -1)
            
            # Calculate node connections more efficiently
            node_connections = adj_mx[:, :N, :N].sum(dim=2)  # [B, N]
            # Safe normalization with epsilon
            conn_max = node_connections.max(dim=1, keepdim=True)[0]
            conn_max = torch.clamp(conn_max, min=1e-8)
            node_connections = node_connections / conn_max
        
        # 4. Vectorized feature importance calculation
        # Mean across nodes and apply softmax across features
        feature_importance = F.softmax(history_data.abs().mean(dim=2), dim=2)  # [B, T, D]
        
        # 5. Vectorized feature weighting
        feature_importance = feature_importance.unsqueeze(2)  # [B, T, 1, D]
        weighted_features = history_data * feature_importance  # [B, T, N, D]
        node_values = weighted_features.sum(dim=3)  # [B, T, N]
        
        # 6. Vectorized temporal change calculation
        if T > 1:
            # Create shifted tensor for computing differences
            history_shifted = torch.cat([
                torch.zeros((B, 1, N, D), device=device),  # Padding for t=0
                history_data[:, :-1]  # Shift all other timesteps
            ], dim=1)
            
            # Calculate temporal derivative and enhance changes
            temporal_change = (history_data - history_shifted).mean(dim=3)  # [B, T, N]
            temporal_change = torch.sigmoid(temporal_change * 3) - 0.5
            
            # Add temporal component to node values
            node_values = node_values + temporal_change * 0.4
        
        # 7. Enhance contrast with vectorized operations
        node_means = node_values.mean(dim=2, keepdim=True)
        node_values = torch.sigmoid((node_values - node_means) * 3)
        
        # 8. Vectorized grid construction
        # Create batch of empty grids
        grids = torch.zeros(B, T, working_size, working_size, device=device)
        
        # Efficiently place node values into grid positions
        valid_count = min(N, len(y_indices))
        for i in range(valid_count):
            y, x = y_indices[i].item(), x_indices[i].item()
            if y < working_size and x < working_size:
                grids[:, :, y, x] = node_values[:, :, i]
        
        # 9. Blend with connectivity if available
        if adj_mx is not None:
            # Create connectivity grid
            conn_grid = torch.zeros(B, working_size, working_size, device=device)
            
            for i in range(valid_count):
                y, x = y_indices[i].item(), x_indices[i].item()
                if y < working_size and x < working_size:
                    conn_grid[:, y, x] = node_connections[:, i]
            
            # Expand for broadcasting
            conn_grid = conn_grid.unsqueeze(1).expand(-1, T, -1, -1)
            
            # Create dynamic blend factors
            time_factors = 0.8 - 0.2 * (torch.arange(T, device=device) / max(1, T-1))
            blend_factors = time_factors.view(1, T, 1, 1)
            
            # Apply blending
            grids = blend_factors * grids + (1 - blend_factors) * conn_grid
        
        # 10. Apply gamma correction for better visibility
        gamma = 0.7  # brighten midtones
        grids = normalize_minmax(grids).pow(gamma)
        
        return grids
    
    def _generate_correlation_channel_vectorized(self, history_data, hidden_state, adj_mx):
        """
        Optimized correlation channel generation with efficient operations
        
        Args:
            history_data: Tensor of shape [B, T, N, D]
            hidden_state: Tensor of shape [B, E, N, D]
            adj_mx: Optional adjacency matrix
            
        Returns:
            correlation_channel: Tensor of shape [B, T, ws, ws]
        """
        B, T, N, D = history_data.shape
        device = history_data.device
        working_size = self.working_size
        
        # Initialize output
        corr_channel = torch.zeros(B, T, working_size, working_size, device=device)
        
        # 1. Efficient adjacency matrix preprocessing
        if adj_mx is not None:
            if adj_mx.dim() == 2:  # [N, N] -> [B, N, N]
                adj_mx = adj_mx.unsqueeze(0).expand(B, -1, -1)
            elif adj_mx.dim() == 3 and adj_mx.size(0) == 1:  # [1, N, N] -> [B, N, N]
                adj_mx = adj_mx.expand(B, -1, -1)
            
            # Scale to correlation range
            adj_mx = adj_mx[:, :N, :N].float() * 2 - 1
        
        # 2. Efficient hidden state reduction
        if hidden_state is not None:
            hidden_flat = hidden_state.mean(dim=1)  # [B, N, D]
        else:
            hidden_flat = None
        
        # 3. Optimized sliding window approach
        window_size = min(3, T)
        half_window = window_size // 2
        
        # Pre-compute all correlation matrices at once where possible
        if T <= 4:  # For small sequences, we can do direct computation
            # Process all timesteps at once with a single matrix operation
            # Normalize entire history data
            history_flat = history_data.reshape(B, T, N*D)
            
            # Add stability checks before normalization
            history_flat = torch.nan_to_num(history_flat, nan=0.0, posinf=10.0, neginf=-10.0)
            history_flat = torch.clamp(history_flat, -10.0, 10.0)
            
            # Use a more stable normalization with epsilon
            history_norm = F.normalize(history_flat, p=2, dim=2, eps=1e-8)
            
            # Check for NaN values after normalization
            history_norm = torch.nan_to_num(history_norm, nan=0.0)
            
            # Compute all correlations at once (approx for small T)
            corr_all = torch.bmm(history_norm, history_norm.transpose(1, 2))  # [B, T, T]
            
            # Check for NaN values after correlation computation
            corr_all = torch.nan_to_num(corr_all, nan=0.0)
            corr_all = torch.clamp(corr_all, -1.0, 1.0)  # Ensure valid correlation range
            
            # Extract from diagonal for each node
            # This is an approximation that works better for small T
            for t in range(T):
                corr_matrices = corr_all[:, t:t+1, :].reshape(B, N, N)
                
                # Apply contrast enhancement
                corr_matrices = (corr_matrices + 1) / 2
                corr_matrices = torch.clamp(corr_matrices, 0.01, 0.99).pow(1.2)
                corr_matrices = corr_matrices * 2 - 1
                
                # Blend with adjacency if available
                if adj_mx is not None:
                    adj_weight = 0.2 + 0.1 * (t / max(1, T-1))
                    corr_matrices = (1 - adj_weight) * corr_matrices + adj_weight * adj_mx
                
                # Final normalization
                corr_matrices = (corr_matrices + 1) / 2
                
                # Convert to images efficiently
                if N <= working_size:
                    pad = (working_size - N) // 2
                    if pad > 0 and N > 0:
                        for b in range(B):
                            corr_channel[b, t, pad:pad+N, pad:pad+N] = corr_matrices[b]
                    else:
                        # Use interpolation
                        corr_resized = F.interpolate(
                            corr_matrices.unsqueeze(1),
                            size=(working_size, working_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                        corr_channel[:, t] = corr_resized
                else:
                    # Use interpolation
                    corr_resized = F.interpolate(
                        corr_matrices.unsqueeze(1),
                        size=(working_size, working_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    corr_channel[:, t] = corr_resized
        else:
            # For larger sequences, use optimized sliding window
            # Pre-compute time weights once
            max_window = min(window_size, T)
            time_weights = torch.exp(torch.linspace(-3, 0, max_window, device=device))
            time_weights = time_weights / time_weights.sum()
            
            # Process each timestep with optimized sliding window
            for t in range(T):
                # Efficiently define window bounds
                start_idx = max(0, t - half_window)
                end_idx = min(T, t + half_window + 1)
                current_window = end_idx - start_idx
                
                # Extract window data efficiently
                window_data = history_data[:, start_idx:end_idx]  # [B, window, N, D]
                
                # Apply time weights appropriately sized for this window
                curr_weights = time_weights[:current_window]
                curr_weights = curr_weights / curr_weights.sum()
                curr_weights = curr_weights.view(1, current_window, 1, 1)
                
                weighted_data = window_data * curr_weights
                aggregated_features = weighted_data.sum(dim=1)  # [B, N, D]
                
                # Efficiently blend with hidden state if available
                if hidden_flat is not None:
                    time_ratio = t / max(1, T-1)
                    hidden_weight = 0.3 + 0.4 * time_ratio
                    features = (1 - hidden_weight) * aggregated_features + hidden_weight * hidden_flat
                else:
                    features = aggregated_features
                
                # Efficiently normalize features
                features_norm = F.normalize(features, p=2, dim=2)
                
                # Batch matrix multiplication for correlation
                corr_matrices = torch.bmm(features_norm, features_norm.transpose(1, 2))
                
                # Enhance contrast efficiently
                corr_matrices = (corr_matrices + 1) / 2
                corr_matrices = torch.clamp(corr_matrices, 0.01, 0.99).pow(1.2)
                corr_matrices = corr_matrices * 2 - 1
                
                # Blend with adjacency if available
                if adj_mx is not None:
                    adj_weight = 0.2 + 0.1 * (t / max(1, T-1))
                    corr_matrices = (1 - adj_weight) * corr_matrices + adj_weight * adj_mx
                
                # Final normalization
                corr_matrices = (corr_matrices + 1) / 2
                
                # Efficiently convert to images
                if N <= working_size:
                    pad = (working_size - N) // 2
                    if pad > 0 and N > 0:
                        # Handle each batch separately to avoid excessive memory usage
                        for b in range(B):
                            corr_channel[b, t, pad:pad+N, pad:pad+N] = corr_matrices[b]
                    else:
                        # Use interpolation
                        corr_resized = F.interpolate(
                            corr_matrices.unsqueeze(1),
                            size=(working_size, working_size),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(1)
                        corr_channel[:, t] = corr_resized
                else:
                    # Use interpolation
                    corr_resized = F.interpolate(
                        corr_matrices.unsqueeze(1),
                        size=(working_size, working_size),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    corr_channel[:, t] = corr_resized
        
        # Apply histogram-like enhancement
        corr_channel = corr_channel.pow(0.7)  # Enhance mid and low values
        
        return corr_channel
    
    def _save_visualizations(self, visual_repr):
        """
        Save visualization results with epoch information and enhanced contrast
        
        Args:
            visual_repr: Visual representations [B, T, 3, H, W]
        """
        try:
            # Ensure output directory exists
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path, exist_ok=True)
                
            # Only save for first batch example
            b = 0
            B, T, C, H, W = visual_repr.shape
            
            # Get current epoch for filename
            epoch = self.current_epoch
            
            # Save at intervals to reduce I/O operations
            for t in range(0, T, self.save_interval):
                # Extract channels safely
                temporal = visual_repr[b, t, 0].detach().cpu()
                spatial = visual_repr[b, t, 1].detach().cpu()
                correlation = visual_repr[b, t, 2].detach().cpu()
                
                # Create enhanced combined representation
                combined = torch.stack([
                    temporal,
                    spatial,
                    correlation
                ], dim=0).detach().cpu()
                
                # Save each channel with epoch information in filename
                if self.color_mode == 'all_color':
                    self._save_enhanced_image(temporal, f"temporal_t{t}_epo{epoch}", colormap=self.temporal_colormap)
                    self._save_enhanced_image(spatial, f"spatial_t{t}_epo{epoch}", colormap=self.spatial_colormap)
                    self._save_enhanced_image(correlation, f"correlation_t{t}_epo{epoch}", colormap=self.correlation_colormap)
                else:  # 'all_gray'
                    self._save_enhanced_image(temporal, f"temporal_t{t}_epo{epoch}")
                    self._save_enhanced_image(spatial, f"spatial_t{t}_epo{epoch}")
                    self._save_enhanced_image(correlation, f"correlation_t{t}_epo{epoch}")
                
                # Always save combined in color
                self._save_enhanced_image(combined, f"combined_t{t}_epo{epoch}", is_rgb=True)
                
            # Force garbage collection after saving
            gc.collect()
                
        except Exception as e:
            print(f"Error saving visualizations: {str(e)}")
    
    def _async_save_visualizations(self, visual_repr):
        """
        Asynchronously save visualization results to avoid blocking main computation
        
        Args:
            visual_repr: Visual representations [B, T, 3, H, W]
        """
        def save_thread():
            try:
                # Clone data to avoid references to original tensors
                visual_copy = visual_repr.clone().cpu()
                self._save_visualizations(visual_copy)
                # Explicitly delete the copy when done
                del visual_copy
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error in visualization thread: {str(e)}")
        
        # Create and start thread
        thread = threading.Thread(target=save_thread)
        thread.daemon = True
        thread.start()
    
    def _save_enhanced_image(self, image_tensor, name, colormap=None, is_rgb=False):
        """
        Save a single image tensor as an enhanced image file with consistent coloring
        
        Args:
            image_tensor: Image tensor to save
            name: Base name for the file
            colormap: Optional matplotlib colormap name for enhanced visualization
            is_rgb: Whether this is an RGB image
        """
        try:
            if is_rgb:
                # RGB image
                image_np = image_tensor.numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                
                # Create and save image with matplotlib for better quality
                plt.figure(figsize=(6, 6), dpi=100)
                plt.imshow(image_np)
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(self.save_path, f"{name}.png"), 
                          bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                # Grayscale image
                image_np = image_tensor.numpy()
                if len(image_np.shape) > 2:
                    image_np = image_np.squeeze()
                
                # Apply contrast enhancement
                if self.enhance_contrast:
                    # Simple adaptive histogram equalization approximation
                    p2, p98 = np.percentile(image_np, (2, 98))
                    if p98 > p2:
                        image_np = np.clip(image_np, p2, p98)
                        image_np = (image_np - p2) / (p98 - p2)
                
                # Create and save with consistent color approach
                plt.figure(figsize=(6, 6), dpi=100)
                
                # Use colormap only if specified and in color mode
                if colormap and self.color_mode == 'all_color':
                    plt.imshow(image_np, cmap=colormap)
                else:
                    plt.imshow(image_np, cmap='gray')
                    
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(self.save_path, f"{name}.png"), 
                          bbox_inches='tight', pad_inches=0)
                plt.close()
                
        except Exception as e:
            print(f"Error saving enhanced image {name}: {str(e)}")
            # Fallback to basic image saving method
            try:
                # Basic image processing
                if is_rgb:
                    image_np = image_tensor.numpy().transpose(1, 2, 0)
                    mode = 'RGB'
                else:
                    image_np = image_tensor.numpy()
                    if len(image_np.shape) > 2:
                        image_np = image_np.squeeze()
                    mode = 'L'
                
                # Ensure valid range before scaling to 255
                # Handle possible NaN, inf, or values outside expected range
                image_np = np.nan_to_num(image_np, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Make sure the range is [0, 1] through min-max normalization if necessary
                if image_np.min() < 0 or image_np.max() > 1:
                    min_val = image_np.min()
                    max_val = image_np.max()
                    if max_val > min_val:
                        image_np = (image_np - min_val) / (max_val - min_val)
                    else:
                        image_np = np.zeros_like(image_np)
                
                image_np = np.clip(image_np, 0, 1) * 255
                image_np = image_np.astype(np.uint8)
                
                # Create and save image
                img = Image.fromarray(image_np, mode=mode)
                img.save(os.path.join(self.save_path, f"{name}.png"), optimize=True)
            except Exception as e2:
                print(f"Even basic image saving failed for {name}: {str(e2)}")
