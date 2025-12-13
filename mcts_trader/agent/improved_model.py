import torch
import torch.nn as nn
from .trm import TinyRecursiveTradingModel
from easydict import EasyDict

class ImprovedTRMMuZeroModel(nn.Module):
    """
    A lean, non-redundant MuZero model for the TRM+MCTS agent.
    Refactored to use a single shared encoder and task-specific heads,
    reusing the recursive reasoning logic from TinyRecursiveTradingModel where appropriate.
    """
    def __init__(self, observation_shape, action_space_size, state_dim=64, categorical_distribution=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_space_size = action_space_size
        self.categorical_distribution = categorical_distribution
        
        # Support size for categorical value/reward
        self.value_support_size = 601 if categorical_distribution else 1
        self.reward_support_size = self.value_support_size # Alias for policy checks

        # 1. Create ONE shared instance of the TRM backbone
        # We will use its encoder and recursive logic
        self.trm = TinyRecursiveTradingModel(
            state_dim=state_dim, 
            n_actions=action_space_size, 
            value_support_size=self.value_support_size
        )

        # 2. Create distinct networks that share the TRM backbone
        self.representation_network = TRMRepresentation(self.trm)
        
        self.dynamics_network = TRMDynamics(
            action_dim=action_space_size, 
            state_dim=state_dim,
            reward_support_size=self.value_support_size
        )

        self.prediction_network = TRMPrediction(self.trm)

    def initial_inference(self, observation):
        """Used for bootstrapping the training data."""
        latent_state = self.representation_network(observation)
        policy_logits, value = self.prediction_network(latent_state)
        
        # Reward at step 0 is 0
        batch_size = observation.shape[0]
        if self.categorical_distribution:
            reward = torch.zeros(batch_size, self.value_support_size).to(observation.device)
            reward[:, 300] = 10.0 # One-hot for 0.0 (assuming center is 300)
        else:
            reward = torch.zeros(batch_size, 1).to(observation.device)
            
        return EasyDict({
            'latent_state': latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        })

    def recurrent_inference(self, latent_state, action):
        """Used for unrolling the dynamics."""
        next_latent_state, reward = self.dynamics_network(latent_state, action)
        policy_logits, value = self.prediction_network(next_latent_state)
        return EasyDict({
            'latent_state': next_latent_state,
            'policy_logits': policy_logits,
            'value': value,
            'reward': reward
        })

class TRMRepresentation(nn.Module):
    """Uses the shared TRM encoder to produce the initial latent state."""
    def __init__(self, trm: TinyRecursiveTradingModel):
        super().__init__()
        self.trm = trm

    def forward(self, x):
        return self.trm.encode(x)

class TRMDynamics(nn.Module):
    """Predicts the next latent state and reward."""
    def __init__(self, action_dim, state_dim=64, reward_support_size=601, hidden_dim=256):
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim), # Takes latent state + embedded action
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim + reward_support_size), # Outputs next state and reward
        )
        self.state_dim = state_dim
        self.reward_support_size = reward_support_size

    def forward(self, latent_state, action):
        # action: (Batch, 1) or (Batch, )
        if action.dim() == 2:
             action = action.squeeze(1)
             
        embedded_action = self.action_embedding(action.long())
        x = torch.cat([latent_state, embedded_action], dim=-1)
        out = self.net(x)
        
        next_state = out[:, :self.state_dim]
        reward = out[:, self.state_dim:]
        return next_state, reward

class TRMPrediction(nn.Module):
    """Uses the shared TRM to predict policy and value from a latent state."""
    def __init__(self, trm: TinyRecursiveTradingModel):
        super().__init__()
        self.trm = trm

    def forward(self, latent_state):
        # Use TRM's recursive logic on the latent state
        # Note: compute_policy_value expects 'state' (which is latent_state here)
        policy_logits, value, _ = self.trm.compute_policy_value(latent_state)
        return policy_logits, value
