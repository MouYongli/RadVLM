"""
Utilities for visualizing attention maps from DeepSeek-VL2 model.
"""
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image


class AttentionVisualizer:
    """Visualize attention maps from transformer models."""
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the attention visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = save_dir
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    def visualize_attention_heatmap(
        self,
        attention: torch.Tensor,
        layer_idx: int = -1,
        head_idx: int = 0,
        title: str = "Attention Heatmap",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        cmap: str = "viridis"
    ) -> plt.Figure:
        """
        Visualize attention weights as a heatmap.
        
        Args:
            attention: Attention tensor of shape (batch, heads, seq_len, seq_len)
                      or (heads, seq_len, seq_len)
            layer_idx: Which layer to visualize (-1 for last)
            head_idx: Which attention head to visualize
            title: Title for the plot
            figsize: Figure size
            save_path: Path to save the figure
            cmap: Colormap to use
            
        Returns:
            matplotlib Figure object
        """
        # Handle different tensor shapes
        if attention.dim() == 4:  # (batch, heads, seq_len, seq_len)
            attention = attention[0]  # Take first batch
        
        if attention.dim() == 3:  # (heads, seq_len, seq_len)
            att_matrix = attention[head_idx].detach().cpu().numpy()
        elif attention.dim() == 2:  # (seq_len, seq_len)
            att_matrix = attention.detach().cpu().numpy()
        else:
            raise ValueError(f"Unexpected attention shape: {attention.shape}")
        
        # Normalize for better visualization
        att_matrix = (att_matrix - att_matrix.min()) / (att_matrix.max() - att_matrix.min())
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(att_matrix, cmap=cmap, ax=ax, cbar_kws={"label": "Attention Weight"})
        ax.set_title(title)
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_to_image(
        self,
        image: Union[Image.Image, str, np.ndarray],
        attention: torch.Tensor,
        head_idx: int = 0,
        token_idx: Optional[int] = None,
        title: str = "Attention on Image",
        figsize: Tuple[int, int] = (12, 5),
        save_path: Optional[str] = None,
        alpha: float = 0.5
    ) -> plt.Figure:
        """
        Overlay attention weights on an image.
        
        Args:
            image: PIL Image, file path, or numpy array
            attention: Attention tensor (assuming last dimension is image patches)
            head_idx: Which attention head to visualize
            token_idx: Which token position to visualize (if None, averages across tokens)
            title: Title for the plot
            figsize: Figure size
            save_path: Path to save the figure
            alpha: Transparency of the attention overlay
            
        Returns:
            matplotlib Figure object
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        
        img_array = np.array(image)
        
        # Extract attention weights
        if attention.dim() == 4:
            attention = attention[0]  # batch
        if attention.dim() == 3:
            attention = attention[head_idx]  # specific head
        
        attention = attention.detach().cpu().numpy()
        
        # If token_idx is None, average attention across all tokens
        if token_idx is None:
            att_weights = attention.mean(axis=0)
        else:
            att_weights = attention[token_idx]
        
        # Normalize
        att_weights = (att_weights - att_weights.min()) / (att_weights.max() - att_weights.min())
        
        # Resize attention to match image size (assuming attention to image patches)
        if att_weights.ndim == 1:
            # Reshape to square for visualization
            side = int(np.sqrt(len(att_weights)))
            if side * side == len(att_weights):
                att_weights = att_weights.reshape(side, side)
            else:
                print(f"Warning: Cannot reshape attention of length {len(att_weights)} to square")
                att_weights = att_weights.reshape(1, -1)
        
        att_weights_resized = np.array(
            Image.fromarray((att_weights * 255).astype(np.uint8)).resize(
                (img_array.shape[1], img_array.shape[0]), Image.Resampling.BILINEAR
            )
        ) / 255.0
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        axes[0].imshow(img_array)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Attention map
        axes[1].imshow(att_weights_resized, cmap='hot')
        axes[1].set_title("Attention Map")
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(img_array)
        axes[2].imshow(att_weights_resized, cmap='hot', alpha=alpha)
        axes[2].set_title("Attention Overlay")
        axes[2].axis('off')
        
        fig.suptitle(title)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_attention_rollout(
        self,
        attentions: List[torch.Tensor],
        title: str = "Attention Rollout",
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize attention rollout across layers.
        
        Args:
            attentions: List of attention tensors from different layers
            title: Title for the plot
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        rollout = torch.eye(attentions[0].shape[-1])
        
        for attention in attentions:
            if attention.dim() == 4:
                attention = attention[0]  # batch
            if attention.dim() == 3:
                attention = attention.mean(dim=0)  # average heads
            
            rollout = torch.matmul(attention, rollout)
        
        rollout = rollout.detach().cpu().numpy()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(rollout, cmap="viridis", ax=ax, cbar_kws={"label": "Rollout Score"})
        ax.set_title(title)
        ax.set_xlabel("Position")
        ax.set_ylabel("Position")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def visualize_multi_head_attention(
        self,
        attention: torch.Tensor,
        layer_idx: int = -1,
        max_heads: int = 8,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize multiple attention heads in a grid.
        
        Args:
            attention: Attention tensor (batch, heads, seq_len, seq_len)
            layer_idx: Which layer to visualize
            max_heads: Maximum number of heads to display
            figsize: Figure size
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if attention.dim() == 4:
            attention = attention[0]  # batch
        
        num_heads = min(attention.shape[0], max_heads)
        cols = 4
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        
        for head_idx in range(num_heads):
            att_matrix = attention[head_idx].detach().cpu().numpy()
            att_matrix = (att_matrix - att_matrix.min()) / (att_matrix.max() - att_matrix.min())
            
            axes[head_idx].imshow(att_matrix, cmap='viridis')
            axes[head_idx].set_title(f"Head {head_idx}")
            axes[head_idx].axis('off')
        
        # Hide unused subplots
        for head_idx in range(num_heads, len(axes)):
            axes[head_idx].axis('off')
        
        fig.suptitle("Multi-Head Attention")
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def create_attention_report(
        self,
        images: List[Union[str, Image.Image, np.ndarray]],
        attentions: Union[torch.Tensor, List[torch.Tensor]],
        study_id: str = "unknown",
        output_prefix: str = None
    ) -> None:
        """
        Create a comprehensive attention visualization report.
        
        Args:
            images: List of images or paths
            attentions: Attention tensor(s) from model
            study_id: Study ID for naming
            output_prefix: Prefix for output files (uses self.save_dir if None)
        """
        if output_prefix is None:
            output_prefix = f"{self.save_dir}/{study_id}" if self.save_dir else study_id
        
        Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
        
        # Handle single vs multiple attention tensors
        if isinstance(attentions, torch.Tensor):
            attentions = [attentions]
        
        # Visualize first attention tensor
        if len(attentions) > 0:
            att = attentions[0]
            
            # Heatmap
            self.visualize_attention_heatmap(
                att,
                save_path=f"{output_prefix}_attention_heatmap.png",
                title=f"Attention Heatmap - {study_id}"
            )
            plt.close('all')
            
            # Multi-head
            self.visualize_multi_head_attention(
                att,
                save_path=f"{output_prefix}_multi_head_attention.png"
            )
            plt.close('all')
            
            # Attention on images
            if images:
                for img_idx, image in enumerate(images[:3]):  # Limit to first 3
                    try:
                        self.visualize_attention_to_image(
                            image,
                            att,
                            save_path=f"{output_prefix}_attention_on_image_{img_idx}.png",
                            title=f"Attention on Image {img_idx} - {study_id}"
                        )
                        plt.close('all')
                    except Exception as e:
                        print(f"Warning: Could not visualize attention on image {img_idx}: {e}")
            
            print(f"Attention report saved with prefix: {output_prefix}")


def visualize_attention_comparison(
    attentions_dict: dict,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Compare attention maps from multiple samples or models.
    
    Args:
        attentions_dict: Dictionary of {label: attention_tensor}
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, len(attentions_dict), figsize=figsize)
    
    for idx, (label, attention) in enumerate(attentions_dict.items()):
        if attention.dim() == 4:
            attention = attention[0, 0]  # batch, head
        elif attention.dim() == 3:
            attention = attention[0]  # head
        
        att_matrix = attention.detach().cpu().numpy()
        att_matrix = (att_matrix - att_matrix.min()) / (att_matrix.max() - att_matrix.min())
        
        axes[idx].imshow(att_matrix, cmap='viridis')
        axes[idx].set_title(label)
        axes[idx].axis('off')
    
    return fig
