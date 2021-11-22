import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.sac.policies import TanhGaussianPolicy


class SingleSAC:
    def __init__(self,
                 obs_dim,
                 latent_dim,
                 action_dim,
                 sac_layer_size):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.sac_layer_size = sac_layer_size

        M = self.sac_layer_size
        self.qf1 = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self.qf2 = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self.target_qf1 = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self.target_qf2 = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self.policy = TanhGaussianPolicy(
            obs_dim=(obs_dim + latent_dim),
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=[M, M, M],
        )
        self.alpha_net = Mlp(
            hidden_sizes=[latent_dim * 10],
            input_size=latent_dim,
            output_size=1
        )

    def forward(self, network, state, task_indicator, base_task_indicator, actions=None, **kwargs):
        if network == 'alpha_net':
            network_input = task_indicator
        else:
            network_input = torch.cat([state, task_indicator], dim=1)

        if network == 'qf1':
            return self.qf1(network_input, actions, **kwargs)
        elif network == 'qf2':
            return self.qf2(network_input, actions, **kwargs)
        elif network == 'target_qf1':
            return self.target_qf1(network_input, actions, **kwargs)
        elif network == 'target_qf2':
            return self.target_qf2(network_input, actions, **kwargs)
        elif network == 'policy':
            return self.policy(network_input, **kwargs)
        elif network == 'alpha_net':
            return self.alpha_net(network_input, **kwargs)
        else:
            raise ValueError(f'{network} is not a valid network specification for {type(self).__name__}.forward(...)')

    def soft_target_update(self, soft_target_tau):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, soft_target_tau
        )

    def get_action(self, state, task_indicator, base_task_indicator, deterministic=False):
        policy_input = torch.cat([state, task_indicator], dim=1)
        return self.policy.get_action(policy_input, deterministic=deterministic)

    def to(self, device, network=None):
        if network is None:
            self.qf1.to(device)
            self.qf2.to(device)
            self.target_qf1.to(device)
            self.target_qf2.to(device)
            self.policy.to(device)
            self.alpha_net.to(device)
        elif network == 'qf1':
            self.qf1.to(device)
        elif network == 'qf2':
            self.qf2.to(device)
        elif network == 'target_qf1':
            self.target_qf1.to(device)
        elif network == 'target_qf2':
            self.target_qf2.to(device)
        elif network == 'policy':
            self.policy.to(device)
        elif network == 'alpha_net':
            self.alpha_net.to(device)
        else:
            raise ValueError(f'{network} is not a valid network specification for {type(self).__name__}.to(...)')

    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
            alpha_net=self.alpha_net
        )

