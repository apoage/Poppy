import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyRecursiveTradingModel(nn.Module):
    """
    Inspired by TRM (Samsung SAIL Montreal, Oct 2025)
    7M params can outperform 671B params LLMs on reasoning tasks
    Adapted for Trading.
    """
    def __init__(self, state_dim=64, hidden_dim=128, n_actions=5, value_support_size=601):
        super().__init__()
        
        # Lightweight encoder
        # Input: (batch, channels, 64, 64)
        # We assume channels=27 roughly from the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(27, 32, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64*4*4, state_dim)
        )
        
        # Recursive reasoning module (the key innovation)
        self.latent_refiner = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # [state, latent] -> hidden
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Output refinement
        self.output_refiner = nn.Sequential(
            nn.Linear(state_dim + n_actions, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions)
        )
        
        # Initial embeddings
        self.initial_latent = nn.Parameter(torch.zeros(state_dim))
        self.initial_output = nn.Parameter(torch.zeros(n_actions))
        
        # Value head (separate from recursive policy)
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, value_support_size)
        )
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        
    def encode(self, x):
        """Encode observation into state"""
        return self.encoder(x)

    def compute_policy_value(self, state, H_cycles=None, L_cycles=None):
        """
        Compute policy and value from state using recursive reasoning.
        """
        batch_size = state.shape[0]
        H = H_cycles if H_cycles is not None else 3
        L = L_cycles if L_cycles is not None else 4
        
        # Initialize latent and output
        latent = self.initial_latent.expand(batch_size, -1)
        output = self.initial_output.expand(batch_size, -1)
        
        # Recursive improvement (TRM core algorithm)
        for h in range(H):
            # Refine latent given state and current output
            for l in range(L):
                combined = torch.cat([state, latent], dim=-1)
                latent = latent + self.latent_refiner(combined)  # Residual
            
            # Refine output given latent
            output_input = torch.cat([latent, output], dim=-1)
            output = output + self.output_refiner(output_input)  # Residual
        
        policy_logits = output
        value = self.value_head(state) # Value based on encoded state
        
        return policy_logits, value, latent

    def forward(self, x, H_cycles=3, L_cycles=4):
        """
        H_cycles: Number of output improvement cycles (K in paper)
        L_cycles: Number of latent refinement cycles (n in paper)
        """
        # Encode state
        state = self.encode(x)
        
        # Compute policy/value
        policy_logits, value, latent = self.compute_policy_value(state, H_cycles, L_cycles)
        
        return policy_logits, value, state, latent

