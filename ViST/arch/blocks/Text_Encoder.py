import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, AutoModel
import os
import time
import hashlib
import math

class TextEncoder(nn.Module):
    """
    Text encoder that handles the complete pipeline from data to text representations
    with clearly separated computational stages:
    1. Generate text prompts from spatio-temporal data
    2. Convert prompts to embeddings using language model
    3. Transform embeddings to text features for downstream tasks
    """
    def __init__(self, config):
        super().__init__()
        # Configuration parameters
        self.config = config
        self.llm_dim = config['llm_dim']
        self.horizon = config.get('horizon', 12)
        self.num_nodes = config['num_nodes']
        self.d_ff = config.get('d_ff', 32)
        self.dropout = config.get('dropout', 0.1)
        self.top_k = config.get('top_k', 5)
        
        # Model setup - lightweight, on-demand loading
        self.model_name = config['llm_model']
        # Support local model path
        if os.path.isdir(self.model_name):
            self._local_model_path = self.model_name
        else:
            self._local_model_path = None
        self.tokenizer = None
        self.llm_model = None
        self._model_loaded = False
        
        # Text feature processor
        self.text_processor = nn.Sequential(
            nn.Linear(self.llm_dim, self.d_ff),
            nn.LayerNorm(self.d_ff),
            nn.GELU()
        )
    
        # Efficient caching for each step
        self._prompt_cache = {}
        self._embedding_cache = {}
        self._feature_cache = {}
        self._cache_size_limit = 100
        self._last_cache_clear = time.time()
        self._cache_ttl = 3600  # Cache time-to-live in seconds
    
    def generate_prompts(self, history_data, data_description=""):
        """
        Generate text prompts from spatio-temporal data statistics
        
        Args:
            history_data: Tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            data_description: Optional dataset description
            
        Returns:
            List of prompt strings
        """
        batch_size, seq_len, num_nodes, input_dim = history_data.shape
        
        # Fast hash-based caching
        try:
            # Use tensor properties instead of full content for faster hashing
            data_hash = f"{history_data[:, -1].mean().item():.3f}_{history_data[:, -1].std().item():.3f}"
            cache_key = f"{data_hash}_{data_description}_{batch_size}"
            
            if cache_key in self._prompt_cache:
                return self._prompt_cache[cache_key]
        except:
            pass

        # Efficient statistics calculation
        with torch.no_grad():
            # Only use last timestep for speed
            x = history_data[:, -1]  # [batch_size, num_nodes, input_dim]
            
            # Fast vectorized operations
            min_vals = x.amin(dim=(1, 2))  # [batch_size]
            max_vals = x.amax(dim=(1, 2))  # [batch_size]
            mean_vals = x.mean(dim=(1, 2))  # [batch_size]
            std_vals = x.std(dim=(1, 2))  # [batch_size]
            
            # Quick node importance calculation
            node_avgs = x.mean(dim=2)  # [batch_size, num_nodes]
            top_vals, top_idxs = torch.topk(node_avgs, k=min(self.top_k, num_nodes), dim=1)
        
        # Generate prompts efficiently
        prompts = []
        for b in range(batch_size):
            # Only include top 3 nodes max to keep prompt small
            top_nodes = ", ".join([f"N{top_idxs[b, i].item()}:{top_vals[b, i].item():.2f}" 
                                  for i in range(min(3, self.top_k))])
            
            # Create a simple prompt with essential information
            prompt = (
                f"Dataset: {data_description} Time: {seq_len-1}: "
                f"min={min_vals[b].item():.2f}, max={max_vals[b].item():.2f}, "
                f"mean={mean_vals[b].item():.2f}, std={std_vals[b].item():.2f}, "
                f"horizon={self.horizon}, top=[{top_nodes}]"
            )
            prompts.append(prompt)
        
        # Cache results
        try:
            self._prompt_cache[cache_key] = prompts
            self._manage_cache_size()
        except:
            pass
            
        return prompts
    
    def _load_language_model(self, device):
        """Lazy-load language model only when needed. Supports BERT and Qwen2.5."""
        if self._model_loaded:
            return True
        try:
            model_path = self._local_model_path if self._local_model_path else self.model_name
            
            # Try to load tokenizer with optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=(self._local_model_path is not None),
                use_fast=True,
                model_max_length=128,
                trust_remote_code=True
            )

            if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                pad_token = '[PAD]'
                self.tokenizer.add_special_tokens({'pad_token': pad_token})
                self.tokenizer.pad_token = pad_token

            # Determine model type based on name
            is_causal_lm = any(k in self.model_name.lower() for k in ['qwen', 'llama', 'gpt', 'mistral'])
            
            if is_causal_lm:
                from transformers import AutoModelForCausalLM
                try:
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        local_files_only=(self._local_model_path is not None),
                        torch_dtype=torch.float16
                    )
                except:
                    print(f"Local model not found. Downloading {self.model_name}...")
                    self.llm_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        local_files_only=False,
                        torch_dtype=torch.float16
                    )
                # Update llm_dim based on actual model hidden size
                self.llm_dim = self.llm_model.config.hidden_size
            else:
                # BERT-style encoder model
                try:
                    self.llm_model = BertModel.from_pretrained(
                        model_path,
                        local_files_only=(self._local_model_path is not None),
                    )
                except:
                    print(f"Local model not found. Downloading {self.model_name}...")
                    self.llm_model = BertModel.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        local_files_only=False
                    )
            
            self._is_causal_lm = is_causal_lm
                
            # Move model to device
            self.llm_model = self.llm_model.to(device)
                
            # Freeze LLM parameters
            for param in self.llm_model.parameters():
                param.requires_grad = False
                
            self._model_loaded = True
            return True
                
        except Exception as e:
            print(f"Warning: Could not load transformer models: {e}")
            self._model_loaded = False
            return False
    
    def get_text_embeddings(self, prompts, device=None):
        """
        Convert text prompts to embeddings using language model with value safety checks
        
        Args:
            prompts: List of prompt strings
            device: Optional device to place tensors on
            
        Returns:
            Tensor of shape [batch_size, llm_dim] with controlled value range
        """
        if device is None and isinstance(prompts, torch.Tensor):
            device = prompts.device
        elif device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model if needed
        model_loaded = self._load_language_model(device)
        
        # Better fallback if model not available - use consistent small values
        if not model_loaded:
            batch_size = len(prompts)
            # Use hash of prompt content for deterministic but stable values
            hash_val = sum(hash(p) % 10000 for p in prompts) / len(prompts)
            torch.manual_seed(int(hash_val))
            # Create small, consistent embeddings within a controlled range
            return torch.randn(batch_size, self.llm_dim, device=device).clamp(-0.5, 0.5) * 0.1
            
        # Process with model
        with torch.no_grad():
            tokens = self.tokenizer(
                prompts,
                padding='max_length',
                max_length=min(128, self.d_ff),
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            if hasattr(self, '_is_causal_lm') and self._is_causal_lm:
                # Causal LM (Qwen2.5, LLaMA, etc.) — use last hidden state
                outputs = self.llm_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask"),
                    output_hidden_states=True
                )
                last_hidden = outputs.hidden_states[-1]
                attention_mask = tokens.get("attention_mask").unsqueeze(-1)
                token_sum = attention_mask.sum(1) + 1e-10
                embeddings = (last_hidden * attention_mask).sum(1) / token_sum
            else:
                # Encoder model (BERT) — use pooled output
                outputs = self.llm_model(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens.get("attention_mask"),
                    return_dict=True
                )
                attention_mask = tokens.get("attention_mask").unsqueeze(-1)
                token_sum = attention_mask.sum(1) + 1e-10
                embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / token_sum
            
            # IMPORTANT: Normalize embeddings to control their scale
            # Using L2 normalization to ensure consistent magnitude
            embeddings = F.normalize(embeddings, p=2, dim=1) * math.sqrt(self.llm_dim)
            
            # Add safety checks for NaN values
            if torch.isnan(embeddings).any():
                # Replace NaNs with zeros
                embeddings = torch.nan_to_num(embeddings, nan=0.0)
            
            # Clip to reasonable range to prevent extreme values
            embeddings = torch.clamp(embeddings, -5.0, 5.0)
        
        return embeddings
    
    def _get_single_embedding(self, prompt, device):
        """Process a single prompt for efficiency"""
        model_loaded = self._load_language_model(device)
        
        if not model_loaded:
            # Deterministic random embedding
            random_seed = int(hash(prompt) % 10000)
            torch.manual_seed(random_seed)
            return torch.randn(1, self.llm_dim, device=device) * 0.1
            
        with torch.no_grad():
            tokens = self.tokenizer(
                prompt,
                padding='max_length',
                max_length=min(128, self.d_ff),
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = self.llm_model(
                input_ids=tokens["input_ids"],
                attention_mask=tokens.get("attention_mask"),
                return_dict=True
            )
            
            attention_mask = tokens.get("attention_mask").unsqueeze(-1)
            embedding = (outputs.last_hidden_state * attention_mask).sum(1) / (attention_mask.sum(1) + 1e-10)
            
        return embedding
    
    def get_text_features(self, text_embeddings):
        """
        Convert text embeddings to stable features for downstream tasks
        
        Args:
            text_embeddings: Tensor of shape [batch_size, llm_dim]
            
        Returns:
            Tensor of shape [batch_size, hidden_dim] with controlled range
        """
        device = text_embeddings.device
        
        # Safety check for input embeddings
        if torch.isnan(text_embeddings).any():
            text_embeddings = torch.nan_to_num(text_embeddings, nan=0.0)
        
        # Clip input embeddings to prevent extreme values
        text_embeddings = torch.clamp(text_embeddings, -5.0, 5.0)
        
        # Process embeddings to features
        text_features = self.text_processor(text_embeddings)  # [batch_size, hidden_dim]
        
        # IMPORTANT: Normalize and scale to control output range
        # This ensures consistent scale regardless of input variations
        norm = torch.norm(text_features, dim=1, keepdim=True) + 1e-8
        text_features = text_features / norm * math.sqrt(self.d_ff / 4)
        
        # Final safety clip
        text_features = torch.clamp(text_features, -3.0, 3.0)
        
        return text_features
        
    def _manage_cache_size(self):
        """Efficiently manage cache size"""
        current_time = time.time()
        
        # Simple TTL-based clearing
        if current_time - self._last_cache_clear > self._cache_ttl:
            self._prompt_cache.clear()
            self._embedding_cache.clear()
            self._feature_cache.clear()
            self._last_cache_clear = current_time
            return
            
        # Keep cache size in check
        if len(self._prompt_cache) > self._cache_size_limit:
            self._prompt_cache = dict(list(self._prompt_cache.items())[-self._cache_size_limit//2:])
            
        if len(self._embedding_cache) > self._cache_size_limit:
            self._embedding_cache = dict(list(self._embedding_cache.items())[-self._cache_size_limit//2:])
            
        if len(self._feature_cache) > self._cache_size_limit:
            self._feature_cache = dict(list(self._feature_cache.items())[-self._cache_size_limit//2:])
    
    def forward(self, history_data=None, prompts=None, text_embeddings=None, data_description=""):
        """
        Complete pipeline from data to text features with multiple entry points.
        Each stage is clearly separated and can be called independently.
        
        Args:
            history_data: Optional tensor of shape [batch_size, seq_len, num_nodes, input_dim]
            prompts: Optional list of prompt strings (if history_data not provided)
            text_embeddings: Optional tensor of shape [batch_size, llm_dim] (if prompts not provided)
            data_description: Optional dataset description for prompt generation
            
        Returns:
            text_features: Tensor of shape [batch_size, hidden_dim]
            prompts: List of prompt strings (if generated from history_data)
            text_embeddings: Tensor of shape [batch_size, llm_dim] (if generated)
        """
        device = None
        
        # Build complete pipeline based on provided entry point
        if history_data is not None:
            device = history_data.device
            # Step 1: Generate prompts from data
            prompts = self.generate_prompts(history_data, data_description)
            
        if text_embeddings is None and prompts is not None:
            # Step 2: Convert prompts to embeddings
            if device is None and isinstance(prompts, torch.Tensor):
                device = prompts.device
            elif device is None and len(prompts) > 0:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            text_embeddings = self.get_text_embeddings(prompts, device)
            
        if text_embeddings is not None:
            # Step 3: Convert embeddings to features
            text_features = self.get_text_features(text_embeddings)
            
            # Return all calculated components
            # This allows the caller to access any intermediate results they need
            results = (text_features,)
            if prompts is not None:
                results += (prompts,)
            if text_embeddings is not None:
                results += (text_embeddings,)
                
            # If only one result, return it directly
            if len(results) == 1:
                return results[0]
            return results
        
        # If we reached here, we didn't have enough inputs
        raise ValueError("Must provide at least one of: history_data, prompts, or text_embeddings")

class TextualOutputHead(nn.Module):
    """
    Processes text features to generate spatio-temporal predictions
    """
    def __init__(self, config):
        super().__init__()
        # Configuration
        self.horizon = config.get('output_len', config.get('horizon', 12))
        self.num_nodes = config['num_nodes']
        self.output_dim = config['output_dim']
        self.hidden_dim = config.get('hidden_dim', 64)
        self.d_ff = config.get('d_ff', 32)
        self.dropout = config.get('dropout', 0.1)
        
        # Node embeddings
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.hidden_dim)
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(self.d_ff, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Output projection - output horizon*output_dim dimensions directly
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.horizon * self.output_dim)
        )
        
        # Cache for inference
        self._cache = {}
        self._cache_size_limit = 50
    
    def forward(self, text_features, hidden_state=None, cache_key=None):
        """
        Generate predictions from text features
        
        Args:
            text_features: [batch_size, d_ff]
            hidden_state: Optional [batch_size, embed_dim, num_nodes, output_dim]
            cache_key: Optional cache key for inference
            
        Returns:
            Predictions [batch_size, horizon, num_nodes, output_dim]
        """
        batch_size = text_features.shape[0]
        device = text_features.device
        
        # No caching - always perform full computation for consistent results
        
        # Transform text features
        text_features = self.feature_transform(text_features)  # [B, hidden_dim]
        
        # CRITICAL: Ensure node embeddings have batch dimension first
        # Expand text features for each node
        text_features = text_features.unsqueeze(1)  # [B, 1, hidden_dim]
        text_features = text_features.expand(batch_size, self.num_nodes, self.hidden_dim)  # [B, N, hidden_dim]
        
        # Get node embeddings
        node_embeddings = self.node_embeddings.unsqueeze(0)  # [1, N, hidden_dim]
        node_embeddings = node_embeddings.expand(batch_size, -1, -1)  # [B, N, hidden_dim]
        
        # Concatenate features
        combined_features = torch.cat([text_features, node_embeddings], dim=2)  # [B, N, hidden_dim*2]
        
        # Generate predictions
        predictions = self.output_projection(combined_features)  # [B, N, horizon*output_dim]
        
        # CRITICAL: Reshape to the EXACT expected output format [B, L, N, C]
        # First reshape to [B, N, L, C]
        predictions = predictions.reshape(batch_size, self.num_nodes, self.horizon, self.output_dim)
        # Then transpose to get [B, L, N, C]
        predictions = predictions.permute(0, 2, 1, 3)  # [B, horizon, num_nodes, output_dim]
        
        # Explicit shape verification with descriptive error
        expected_shape = (batch_size, self.horizon, self.num_nodes, self.output_dim)
        if predictions.shape != expected_shape:
            raise ValueError(f"Output shape error! Expected {expected_shape}, got {predictions.shape}")
        
        return predictions