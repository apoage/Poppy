from typing import Optional, Tuple
import torch
import torch.nn as nn
from lzero.model.muzero_model import MuZeroModel
from .trm import TinyRecursiveTradingModel
from easydict import EasyDict

class TRMRepresentation(nn.Module):
    def __init__(self, trm):
        super().__init__()
        self.trm = trm
    def forward(self, x):
        return self.trm.encode(x)

class TRMPrediction(nn.Module):
    def __init__(self, trm):
        super().__init__()
        self.trm = trm
    def forward(self, x):
        policy, value, _ = self.trm.compute_policy_value(x)
        return {'policy': policy, 'value': value}

class TRMDynamics(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256, reward_support_size=601):
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim + reward_support_size) # Next state + Reward logits
        )
        self.state_dim = state_dim
        
    def forward(self, state, action_encoding):
        # action_encoding: (Batch, 1) or (Batch, )
        if action_encoding.dim() == 2:
             action_encoding = action_encoding.squeeze(1)
             
        action_emb = self.action_embedding(action_encoding.long())
             
        x = torch.cat([state, action_emb], dim=-1)
        out = self.net(x)
        next_state = out[:, :self.state_dim]
        reward = out[:, self.state_dim:]
        return {'next_state': next_state, 'reward': reward}

class TRMMuZeroModel(MuZeroModel):
    """
    LightZero-compatible MuZero model wrapping the TRM architecture.
    """
    def __init__(self, 
                 observation_shape: Tuple[int, int, int] = (27, 64, 64), 
                 action_space_size: int = 320,
                 state_dim: int = 64,
                 categorical_distribution: bool = True,
                 *args, **kwargs):
        # Initialize parent without creating default networks
        super().__init__(observation_shape, action_space_size, *args, **kwargs)
        
        self.categorical_distribution = categorical_distribution
        # Default support size in lzero is 601
        support_size = 601 if categorical_distribution else 1
        
        # Override the networks with TRM-based ones
        self.trm = TinyRecursiveTradingModel(state_dim=state_dim, n_actions=action_space_size, value_support_size=support_size)
        
        # 1. Representation
        self.representation_network = TRMRepresentation(self.trm)
        
        # 2. Dynamics (New simple MLP)
        self.dynamics_network = TRMDynamics(state_dim, action_space_size, reward_support_size=support_size)
        
        # 3. Prediction
        self.prediction_network = TRMPrediction(self.trm)

    def initial_inference(self, obs):
        encoded_state = self.representation_network(obs)
        pred = self.prediction_network(encoded_state)
        
        # Reward at step 0 is 0. Return logits peaking at 0 (index 300 for support 601)
        batch_size = obs.shape[0]
        if self.categorical_distribution:
            reward = torch.zeros(batch_size, 601).to(obs.device)
            reward[:, 300] = 10.0
        else:
            reward = torch.zeros(batch_size, 1).to(obs.device)
            
        return EasyDict({
            'latent_state': encoded_state,
            'policy_logits': pred['policy'],
            'value': pred['value'],
            'reward': reward
        })

    def recurrent_inference(self, latent_state, action):
        dyn = self.dynamics_network(latent_state, action)
        next_latent = dyn['next_state']
        reward = dyn['reward']
        pred = self.prediction_network(next_latent)
        return EasyDict({
            'latent_state': next_latent,
            'policy_logits': pred['policy'],
            'value': pred['value'],
            'reward': reward
        })
