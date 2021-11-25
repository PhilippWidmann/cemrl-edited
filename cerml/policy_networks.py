from abc import ABC, abstractmethod
import torch
import numpy as np
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.sac.policies import TanhGaussianPolicy


class SACNetworks(ABC):
    @abstractmethod
    def forward(self, network, state, task_indicator, base_task_indicator, actions=None, **kwargs):
        pass

    @abstractmethod
    def soft_target_update(self, soft_target_tau):
        pass

    @abstractmethod
    def get_action(self, state, task_indicator, base_task_indicator, deterministic=False):
        pass

    @abstractmethod
    def to(self, device, network=None):
        pass

    @abstractmethod
    def get_networks(self, network=None):
        pass

    @abstractmethod
    def get_network_parameters(self, network):
        pass


class SingleSAC(SACNetworks):
    def __init__(self,
                 obs_dim,
                 latent_dim,
                 action_dim,
                 sac_layer_size):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.sac_layer_size = sac_layer_size
        self._networks = dict()

        M = self.sac_layer_size
        self._networks['qf1'] = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self._networks['qf2'] = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self._networks['target_qf1'] = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self._networks['target_qf2'] = FlattenMlp(
            input_size=(obs_dim + latent_dim) + action_dim,
            output_size=1,
            hidden_sizes=[M, M, M],
        )
        self._networks['policy'] = TanhGaussianPolicy(
            obs_dim=(obs_dim + latent_dim),
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_sizes=[M, M, M],
        )
        self._networks['alpha_net'] = Mlp(
            hidden_sizes=[latent_dim * 10],
            input_size=latent_dim,
            output_size=1
        )

    def forward(self, network, state, task_indicator, base_task_indicator, actions=None, **kwargs):
        if network == 'alpha_net':
            network_input = task_indicator
        else:
            network_input = torch.cat([state, task_indicator], dim=1)

        if network in ['qf1', 'qf2', 'target_qf1', 'target_qf2']:
            return self._networks[network].forward(network_input, actions, **kwargs)
        elif network in ['policy', 'alpha_net']:
            return self._networks[network].forward(network_input, **kwargs)
        else:
            raise ValueError(f'{network} is not a valid network specification for {type(self).__name__}.forward(...)')

    def soft_target_update(self, soft_target_tau):
        ptu.soft_update_from_to(
            self._networks['qf1'], self._networks['target_qf1'], soft_target_tau
        )
        ptu.soft_update_from_to(
            self._networks['qf2'], self._networks['target_qf2'], soft_target_tau
        )

    def get_action(self, state, task_indicator, base_task_indicator, deterministic=False):
        policy_input = torch.cat([state, task_indicator], dim=1)
        return self._networks['policy'].get_action(policy_input, deterministic=deterministic)

    def to(self, device, network=None):
        if network is None:
            for net in self._networks.values():
                net.to(device)
        else:
            self._networks[network].to(device)

    def get_networks(self, network=None):
        if network is None:
            return self._networks
        else:
            return self._networks[network]

    def get_network_parameters(self, network):
        return self._networks[network].parameters()


class MultipleSAC(SACNetworks):
    def __init__(self,
                 obs_dim,
                 latent_dim,
                 action_dim,
                 sac_layer_size,
                 num_networks):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.sac_layer_size = sac_layer_size
        self.num_networks = num_networks

        self._single_SACs = [SingleSAC(obs_dim, latent_dim, action_dim, sac_layer_size) for i in range(num_networks)]

    def forward(self, network, state, task_indicator, base_task_indicator, actions=None, **kwargs):
        output_list = []
        network_indicator_list = []
        for k in range(self.num_networks):
            network_indicator = base_task_indicator == k
            if network_indicator.sum() == 0:
                continue
            network_indicator_list.append(network_indicator)
            output_list.append(self._single_SACs[k].forward(network,
                                                            state[network_indicator],
                                                            task_indicator[network_indicator],
                                                            base_task_indicator[network_indicator],
                                                            actions[network_indicator] if actions is not None else None,
                                                            **kwargs))

        if network == 'policy':
            output = []
            for i in range(len(output_list[0])):
                if output_list[0][i] is None:
                    output_element = None
                else:
                    output_element = torch.zeros((state.shape[0], *output_list[0][i].shape[1:]),
                                                 device=output_list[0][i].device)
                    for k in range(self.num_networks):
                        output_element[network_indicator_list[k]] = output_list[k][i]
                output.append(output_element)
        else:
            output = torch.zeros((state.shape[0], *output_list[0].shape[1:]), device=output_list[0].device)
            for k in range(self.num_networks):
                output[network_indicator_list[k]] = output_list[k]

        return output

    def soft_target_update(self, soft_target_tau):
        for k in range(self.num_networks):
            self._single_SACs[k].soft_target_update(soft_target_tau)

    def get_action(self, state, task_indicator, base_task_indicator, deterministic=False):
        action_list = []
        network_indicator_list = []
        for k in range(self.num_networks):
            network_indicator = base_task_indicator == k
            if network_indicator.sum() == 0:
                continue
            network_indicator_list.append(network_indicator.cpu().numpy())
            action = self._single_SACs[k].get_action(state[network_indicator],
                                                               task_indicator[network_indicator],
                                                               base_task_indicator[network_indicator],
                                                               deterministic)[0]
            if len(action.shape) == 1:
                action = np.expand_dims(action, axis=0)
            action_list.append(action)
        actions = np.zeros((state.shape[0], *action_list[0].shape[1:]), dtype=action_list[0].dtype)
        for k in range(len(action_list)):
            actions[network_indicator_list[k]] = action_list[k]

        return actions.squeeze(), {}

    def to(self, device, network=None):
        for k in range(self.num_networks):
            self._single_SACs[k].to(device, network)

    def get_networks(self, network=None):
        res = {}
        for k in range(self.num_networks):
            res.update({f'sac{k}_{name}': net for name, net in self._single_SACs[k].get_networks().items()})
        return res

    def get_network_parameters(self, network):
        modules = []
        for k in range(self.num_networks):
            modules.append(self._single_SACs[k].get_networks(network))
        return torch.nn.ModuleList(modules).parameters()

