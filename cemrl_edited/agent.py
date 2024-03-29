import torch
import torch.nn as nn
import numpy as np
from rlkit.torch.core import np_ify
import rlkit.torch.pytorch_util as ptu

from cemrl_edited.scripted_policies import policies

class CEMRLAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy
                 ):
        super(CEMRLAgent, self).__init__()
        self.encoder = encoder
        self.prior_pz = prior_pz
        self.policy = policy

    def get_action(self, encoder_input, state, input_padding=None, deterministic=False, z_debug=None, env=None,
                   return_distributions=False, agent_info=None):
        state = ptu.from_numpy(state).view(1, -1)
        if return_distributions:
            z, y, distribution = self.encoder(encoder_input, return_distributions=return_distributions, padding_mask=input_padding)
        else:
            z, y = self.encoder(encoder_input, padding_mask=input_padding)
        if z_debug is not None:
            z = z_debug
        a, a_info = self.policy.get_action(state, z, y, deterministic=deterministic)
        a_info['exploration_trajectory'] = False
        if return_distributions:
            a_info['latent_distribution'] = distribution
        return a, a_info, \
               np_ify(z.clone().detach())[0, :], \
               np_ify(y.clone().detach())[0]


class ScriptedPolicyAgent(nn.Module):
    def __init__(self,
                 encoder,
                 prior_pz,
                 policy
                 ):
        super(ScriptedPolicyAgent, self).__init__()
        self.latent_dim = encoder.latent_dim

    def get_action(self, encoder_input, state, deterministic=False, z_debug=None, env=None):
        env_name = env.active_env_name
        oracle_policy = policies[env_name]()
        action = oracle_policy.get_action(state)
        return (action.astype('float32'), {}), \
               np.zeros(self.latent_dim, dtype='float32'), \
               np.zeros(1, dtype='float32')
