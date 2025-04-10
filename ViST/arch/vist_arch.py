import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from .blocks.MLP import MultiLayerPerceptron
from .blocks.utils import *
from .blocks.Text_Encoder import *
from .blocks.MultimodalFusion import *  
from .blocks.MultiPerspectiveVisualEncoder import *

class BlockWiseCrossAttention(nn.Module):
    def __init__(self, embed_dim, vis_dim, dropout=0.1):
        super(BlockWiseCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.dropout = dropout
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.scale_factor = math.sqrt(embed_dim)
    
    def _calculate_block_config(self, seq_length):
        block_count = max(1, int(math.ceil(seq_length ** (2/3))))
        block_size = int(math.ceil(seq_length / block_count))
        return block_size, block_count

    def forward(self, state1, state2):
        batch_size, seq_len, _ = state1.shape
        device = state1.device
        
        block_size, num_blocks = self._calculate_block_config(seq_len)
        
        enhanced_state1_blocks = []
        enhanced_state2_blocks = []
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, seq_len)
            
            if start_idx >= seq_len:
                break
                
            block1 = state1[:, start_idx:end_idx, :]
            block2 = state2[:, start_idx:end_idx, :]
            
            q1 = self.q_proj(block1)
            k2 = self.k_proj(block2)
            v2 = self.v_proj(block2)
            
            q2 = self.q_proj(block2)
            k1 = self.k_proj(block1)
            v1 = self.v_proj(block1)
            
            scores12 = torch.bmm(q1, k2.transpose(1, 2)) / self.scale_factor
            attn12 = F.softmax(scores12, dim=-1)
            attn12 = self.dropout_layer(attn12)
            enhanced_block1 = torch.bmm(attn12, v2)
            
            # Attention from state2 to state1
            scores21 = torch.bmm(q2, k1.transpose(1, 2)) / self.scale_factor
            attn21 = F.softmax(scores21, dim=-1)
            attn21 = self.dropout_layer(attn21)
            enhanced_block2 = torch.bmm(attn21, v1)
            
            enhanced_state1_blocks.append(enhanced_block1)
            enhanced_state2_blocks.append(enhanced_block2)
        
        enhanced_state1 = torch.cat(enhanced_state1_blocks, dim=1)
        enhanced_state2 = torch.cat(enhanced_state2_blocks, dim=1)
        
        return enhanced_state1, enhanced_state2


class CrossModalFusionLayer(nn.Module):
    def __init__(self, embed_dim, vis_dim, num_heads=8, dropout=0.1, block_wise=True):
        super(CrossModalFusionLayer, self).__init__()
        self.embed_dim = embed_dim
        self.vis_dim = vis_dim
        self.block_wise = block_wise
        
        self.norm = nn.LayerNorm(embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)
        self.vis_projection = nn.Linear(vis_dim, embed_dim)

        if block_wise: # use block wise cross attention
            self.attn = BlockWiseCrossAttention(embed_dim, vis_dim, dropout)
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.importance_estimator = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        self.ff_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        self.use_amp = True
    
    def forward(self, hidden_embed, visual_repr):
        batch_size, embed_dim, num_nodes, output_dim = hidden_embed.shape
        device = hidden_embed.device
        
        hidden_repr = hidden_embed.permute(0, 2, 3, 1).reshape(batch_size, num_nodes*output_dim, embed_dim)
        vis_repr = visual_repr.permute(0, 2, 3, 1).reshape(batch_size, num_nodes*output_dim, self.vis_dim)
        
        # Project visual representation to match embedding dimension
        vis_repr_proj = self.vis_projection(vis_repr)
        
        hidden_repr_norm = self.norm(hidden_repr)
        vis_repr_norm = self.norm(vis_repr_proj)
        
        if self.block_wise:
            # Both inputs to BlockWiseCrossAttention must have same embedding dimension
            tmp_repr, vis_repr = self.attn(hidden_repr_norm, vis_repr_norm)
        else:
            tmp_repr, _ = self.attn(
                query=hidden_repr_norm,
                key=vis_repr_norm,
                value=vis_repr_norm
            )
            
            vis_repr, _ = self.attn(
                query=vis_repr_norm,
                key=hidden_repr_norm,
                value=hidden_repr_norm
            )
            
        global_tmp = tmp_repr.mean(dim=1)
        global_vis = vis_repr.mean(dim=1)
        global_concat = torch.cat([global_tmp, global_vis], dim=1)
        
        importance = self.importance_estimator(global_concat)
        
        tmp_weight = importance[:, 0].unsqueeze(1).unsqueeze(2)
        vis_weight = importance[:, 1].unsqueeze(1).unsqueeze(2) 
        
        fused_repr = tmp_weight * tmp_repr + vis_weight * vis_repr
        
        fused_repr = self.fusion_norm(fused_repr)
        fused_output = fused_repr + self.ff_network(fused_repr)
        
        fused_output = fused_output.reshape(batch_size, num_nodes, output_dim, embed_dim)
        fused_output = fused_output.permute(0, 3, 1, 2)
        
        return fused_output

class CVFR(nn.Module): #Conditional Visiual Feature Reconstructor
    def __init__(self, config):
        super().__init__()
        self.horizon = config['horizon']
        self.num_nodes = config['num_nodes']
        self.output_dim = config['output_dim']
        self.hidden_dim = config['hidden_dim']
        self.vis_dim = config['d_vis']
        self.dropout = config['dropout']
        self.image_size = config['image_size']
        self.scale = 1 #config['scale']
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, self.hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU()
        )
        
        # Global pooling to create compact feature representation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal feature extractor to capture sequence patterns
        self.GRU_extractor = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.structural_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_nodes * self.vis_dim)
        )
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.hidden_dim + config.get('d_ff', 32), self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Output dimension mapping
        self.output_projection = nn.Linear(self.vis_dim, self.output_dim)
    
    def forward(self, visual_input, adj_mx=None, text_features=None, cache_key=None):
        batch_size = visual_input.shape[0]
        seq_len = visual_input.shape[1]
        device = visual_input.device

        # Reshape input [B, T, 3, H, W] -> [B*T, 3, H, W]
        vis_batched = visual_input.reshape(-1, visual_input.shape[2], 
                                           visual_input.shape[3], visual_input.shape[4])
        
        vis_batched = self.visual_encoder(vis_batched)  # [B*T, hidden_dim, h, w]
        
        pooled_features = self.global_pool(vis_batched).squeeze(-1).squeeze(-1)  # [B*T, hidden_dim]
        pooled_features = pooled_features.reshape(batch_size, seq_len, -1)
        
        temporal_features, _ = self.GRU_extractor(pooled_features)
        
        final_features = temporal_features[:, -1]  # [B, hidden_dim]

        # Text Feature Generator (TFG)
        if text_features is not None:
            C_t = text_features.clone()
            if final_features.size(0) != text_features.size(0):
                if final_features.size(0) < text_features.size(0):
                    C_t = text_features[:final_features.size(0)]
                else:
                    C_t = text_features[0:1].expand(final_features.size(0), -1)
            
            combined_features = torch.cat([final_features, C_t], dim=1)
            final_features = self.semantic_proj(combined_features)
        
        # Node-level projection for graph structure
        # move behind MSGC?
        #visual_reconstructed = self.structural_proj(final_features)  # [B, num_nodes * vis_dim]
        #visual_reconstructed = visual_reconstructed.reshape(batch_size, self.num_nodes, self.vis_dim)
        
        # Multi-scale Graph Convolution (MSGC)
        if adj_mx is not None:
            if not isinstance(adj_mx, torch.Tensor):
                adj = torch.tensor(adj_mx, device=device).float()
            else:
                adj = adj_mx.float()
                
            # Normalize adjacency matrix
            row_sum = adj.sum(dim=-1, keepdim=True)
            C_g = adj / (row_sum + 1e-8)

            # we need to capture multi-scale information -> GNN
            # for i in range(self.scale):
            #     C_g = torch.bmm(normalized_adj.expand(batch_size, -1, -1), C_g)
        
            final_features = torch.bmm(C_g.expand(batch_size, -1, -1), final_features)

        visual_reconstructed = self.structural_proj(final_features)  # [B, num_nodes * vis_dim]
        visual_reconstructed = visual_reconstructed.reshape(batch_size, self.num_nodes, self.vis_dim)

        visual_reconstructed = visual_reconstructed.permute(0, 2, 1).unsqueeze(-1)  # [B, vis_dim, num_nodes, 1]
        visual_reconstructed = visual_reconstructed.expand(-1, -1, -1, self.output_dim)  # [B, vis_dim, num_nodes, output_dim]
        
        return visual_reconstructed

class STIM(nn.Module):
    def __init__(self, seq_len, input_dim, tem_dim, output_dim, num_nodes, 
                 node_dim=32, d_temp_tid=32, d_temp_diw=32, 
                 time_of_day_size=288, day_of_week_size=7):
        super().__init__()
        # Basic dimensions
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = tem_dim
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        
        # Temporal embedding dimensions
        self.d_temp_tid = d_temp_tid
        self.d_temp_diw = d_temp_diw
        self.time_of_day_size = time_of_day_size
        self.day_of_week_size = day_of_week_size

        # Spatial embeddings
        self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        nn.init.xavier_uniform_(self.node_emb)
            
        # Temporal embeddings
        self.time_in_day_emb = nn.Parameter(torch.empty(self.time_of_day_size, self.d_temp_tid))
        nn.init.xavier_uniform_(self.time_in_day_emb)
            
        self.day_in_week_emb = nn.Parameter(torch.empty(self.day_of_week_size, self.d_temp_diw))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        # Time series embedding layer
        self.adp_emb_layer = nn.Conv2d(
            in_channels=input_dim * seq_len, 
            out_channels=self.embed_dim, 
            kernel_size=(1, 1), 
            bias=True
        )
        
        # Calculate total hidden dimension
        self.hidden_dim = self.embed_dim + self.node_dim + self.d_temp_tid + self.d_temp_diw
        
        # Projection to output dimension
        self.output_projection = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=tem_dim * output_dim,
            kernel_size=(1, 1),
            bias=True
        )
    
    def forward(self, x):
        """
        Transform input tensor to target dimensions
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            
        Returns:
            Tensor of shape [batch_size, tem_dim, num_nodes, output_dim]
        """
        # Prepare data - assuming first channels are the main features
        batch_size, seq_len, num_nodes, _ = x.shape
        input_data = x[..., :self.input_dim]
        
        # Time series embedding
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        adp_emb = self.adp_emb_layer(input_data)  # [B, embed_dim, N, 1]
        
        # Node embeddings - spatial embedding
        node_emb = []
        node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        
        # Temporal embeddings
        tem_emb = []
        t_i_d_data = x[..., 1]  # Assuming time-of-day is channel 1
        time_in_day_emb = self.time_in_day_emb[
        (t_i_d_data[:, -1, :] * self.time_of_day_size).type(torch.LongTensor)]
        tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
            
        d_i_w_data = x[..., 2]  # Assuming day-of-week is channel 2
        day_in_week_emb = self.day_in_week_emb[
            (d_i_w_data[:, -1, :] * self.day_of_week_size).type(torch.LongTensor)]
        tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        # Concatenate all embeddings
        hidden_state = torch.cat([adp_emb] + node_emb + tem_emb, dim=1)  # [B, hidden_dim, N, 1]
        
        # Project to output dimensions
        output = self.output_projection(hidden_state)  # [B, tem_dim*output_dim, N, 1]
        output = output.squeeze(-1).view(batch_size, self.embed_dim, self.output_dim, num_nodes)
        output = output.permute(0, 1, 3, 2)  # [B, tem_dim, N, output_dim]
        
        return output

class ViST(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        
        self.config = {
            'num_nodes': model_args['num_nodes'],
            'input_len': model_args['input_len'],
            'horizon': model_args['output_len'],
            'input_dim': model_args['input_dim'],
            'output_dim': model_args['output_dim'],
            'd_temp': model_args.get('d_temp', 512),
            'hidden_dim': model_args.get('hidden_dim', 256),
            'only_visual': model_args.get('only_visual', False),
            
            'image_size': model_args.get('image_size', 64),
            'interpolation': model_args.get('interpolation', 'bilinear'),
            'save_images': model_args.get('save_images', False),
            'd_vis': model_args.get('d_vis', 512),

            'vocab_size': model_args.get('vocab_size', 10000),
            'llm_dim': model_args.get('llm_dim', 768),
            'llm_model': model_args.get('llm_model', 'bert-base-uncased'),
            'd_ff': model_args.get('d_ff', 32),
            'n_heads': model_args.get('n_heads', 8),
            'top_k': model_args.get('top_k', 3),
            
            'dropout': model_args.get('dropout', 0.1),
            'encoder_layers': model_args.get('encoder_layers', 3),
            'd_conf': model_args.get('d_conf', 256),
            'output_type': model_args.get('output_type', "full"),
            'save_interval': model_args.get('save_interval', 1),
            'use_amp': model_args.get('use_amp', True),
        }
        
        self.output_type = self.config['output_type']
        self.only_visual = self.config['only_visual']
        self.seq_len = self.config['input_len']
        self.horizon = self.config['horizon']
        self.num_nodes = self.config['num_nodes']
        self.domain = model_args.get('data_description', "SD")
        try:
            self.data_description = load_domain_text(self.domain)
        except:
            self.data_description = ""
            
        self.config['description'] = self.data_description
        
        self.mvt = MultiPerspectiveVisualEncoder(self.config)
        self.text_encoder = TextEncoder(self.config)
        self.textual_output_head = TextualOutputHead(self.config)
        
        # Temporal mapping and encoder
        self.st_identity_mapping = STIM(
            seq_len=self.config['input_len'], 
            input_dim=self.config['input_dim'], 
            tem_dim=self.config['d_temp'],
            output_dim=self.config['output_dim'], 
            num_nodes=self.config['num_nodes']
        )
        
        self.mlp_preditor = nn.Sequential(
            *[MultiLayerPerceptron(self.config['horizon'], self.config['horizon']) 
              for _ in range(self.config['encoder_layers'])]
        )

        self.temporal_output_head = nn.Conv2d(
            in_channels=self.config['d_temp'], 
            out_channels=self.config['horizon'], 
            kernel_size=(1, 1), 
            bias=True
        )

        self.conv_sampler = nn.Conv2d(
            in_channels=self.config['d_temp']+self.config['d_vis'], 
            out_channels=self.config['horizon'], 
            kernel_size=(1, 1), 
            bias=True
        )

        # Visual output head
        self.reconstructor = CVFR(self.config) 

        # Cross-modal fusion
        self.cross_modal_fusion = CrossModalFusionLayer(
            embed_dim=self.config['d_temp']+self.config['d_vis'],
            vis_dim=self.config['d_vis'],
            num_heads=self.config['n_heads'],
            dropout=self.config['dropout']
        )

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, history_data, future_data=None, adj_mx=None, batch_seen=None, epoch=None, **kwargs):
        if torch.isnan(history_data).any():
            #print(f"Warning: NaN values detected in input history_data (epoch {epoch})")
            history_data = torch.nan_to_num(history_data, nan=0.0)
        
        B, T, N, D = history_data.shape
        hidden_state = self.st_identity_mapping(history_data)  # [batch_size, embed_dim, num_nodes, output_dim]
        
        adj_tensor = None
        if adj_mx is not None:
            if not isinstance(adj_mx, torch.Tensor):
                adj_tensor = torch.tensor(adj_mx, device=history_data.device)
            else:
                adj_tensor = adj_mx
        
        # for image saving
        if hasattr(self.mvt, 'set_epoch') and epoch is not None:
            self.mvt.set_epoch(epoch)
        
        vis_repr_trans = self.mvt(
            hidden_state.clone(),
            history_data=history_data,
            adj_mx=adj_tensor,
            save_images=self.config['save_images'] and epoch is not None and (batch_seen is None or batch_seen % 100 == 0)
        )

        vis_repr_trans = torch.nan_to_num(vis_repr_trans, nan=0.0)
        vis_repr_trans = torch.clamp(vis_repr_trans, -10.0, 10.0)
        
        if self.output_type == "without_conditions":
            # fix: there's not ViST-V(only_visual), it's without conditions(Multi-modal Conditional Reconstruction）
            vis_embedding = self.reconstructor(
                visual_input=vis_repr_trans,
                adj_mx=None,
                text_features=None
            )
            # Ensure visual embedding doesn't contain NaN values
            vis_embedding = torch.nan_to_num(vis_embedding, nan=0.0)
            vis_embedding = torch.clamp(vis_embedding, -10.0, 10.0)
            
            if self.only_visual:
                hidden_state = torch.zeros_like(hidden_state)
            fused_embedding = torch.cat([hidden_state, vis_embedding], dim=1)
            output = self.conv_sampler(fused_embedding)
            output = self.mlp_preditor(output)  # Add MLP predictor to stabilize output
            return output
            
        elif self.output_type == "only_temporal":
            output = self.temporal_output_head(hidden_state)
            return output
        
        elif self.output_type == "only_textual": # we won't use this
            prompts = self.text_encoder.generate_prompts(history_data, self.data_description)
            text_representation = self.text_encoder(prompts=prompts)

            if isinstance(text_representation, tuple):
                text_features = text_representation[0]
            else:
                text_features = self.text_encoder.get_text_features(text_representation)
                
            output = self.textual_output_head(text_features, hidden_state)
            return output
        
        # same as only_textual to get text features
        prompts = self.text_encoder.generate_prompts(history_data, self.data_description)
        text_representation = self.text_encoder(prompts=prompts)

        if isinstance(text_representation, tuple):
            text_features = text_representation[0]
        else:
            text_features = self.text_encoder.get_text_features(text_representation)
        
        if self.output_type == "without_adj":
            adj_mx = None
        elif self.output_type == "without_text":
            text_features = None
        
        visual_state = self.reconstructor(
            visual_input=vis_repr_trans,
            adj_mx=adj_tensor,
            text_features=text_features
        )
        
        fused_embedding = torch.cat([hidden_state, visual_state], dim=1)

        fused_state = self.cross_modal_fusion(
            hidden_embed=fused_embedding,
            visual_repr=visual_state
        )
        
        output = self.conv_sampler(fused_state)
        
        output = self.mlp_preditor(output)

        return output