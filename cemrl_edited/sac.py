# This code is based on rlkit sac_v2 implementation.

from collections import OrderedDict
import os

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
import gtimer as gt

import rlkit.torch.pytorch_util as ptu
from cemrl_edited.policy_networks import SACNetworks
from rlkit.core.eval_util import create_stats_ordered_dict

import matplotlib.pyplot as plt


class PolicyTrainer:
    def __init__(
            self,
            policy_networks: SACNetworks,

            replay_buffer,
            encoder,
            batch_size,

            env_action_space,
            data_usage_sac,
            use_data_normalization,
            use_sac_data_normalization,
            sac_uses_exploration_data,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=3e-4,
            qf_lr=3e-4,
            optimizer_class=optim.Adam,

            soft_target_tau=5e-3,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            use_parametrized_alpha=False,
            target_entropy=None,
            target_entropy_factor=1.0,
            alpha=1.0

    ):
        super().__init__()
        self.policy_networks = policy_networks
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.replay_buffer = replay_buffer
        self.encoder = encoder
        self.batch_size = batch_size

        self.env_action_space = env_action_space
        self.data_usage_sac = data_usage_sac
        self.use_data_normalization = use_data_normalization
        self.use_sac_data_normalization = use_sac_data_normalization
        self.sac_uses_exploration_data = sac_uses_exploration_data

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.use_parametrized_alpha = use_parametrized_alpha
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -self.env_action_space  # heuristic value from Tuomas
            self.target_entropy = self.target_entropy * target_entropy_factor

            if self.use_parametrized_alpha:
                self.alpha_optimizer = optimizer_class(
                    self.policy_networks.get_network_parameters('alpha_net'),
                    lr=policy_lr,
                )
            else:
                self.log_alpha = ptu.zeros(1, requires_grad=True)
                self.alpha_optimizer = optimizer_class(
                    [self.log_alpha],
                    lr=policy_lr,
                )
        self._alpha = alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy_networks.get_network_parameters('policy'),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.policy_networks.get_network_parameters('qf1'),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.policy_networks.get_network_parameters('qf2'),
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True

    def train(self, epochs):
        gt.stamp('pt_train_start')
        indices = np.array(self.replay_buffer.get_allowed_points(include_exploration=self.sac_uses_exploration_data))
        if self.data_usage_sac == 'tree_sampling':
            indices = np.random.permutation(indices)
        policy_losses = []
        alphas = []
        log_pis = []
        for epoch in range(epochs):
            policy_loss, alpha, log_pi = self.training_step(indices, epoch)
            policy_losses.append(policy_loss/1.0)
            alphas.append(alpha / 1.0)
            log_pis.append((-1) * log_pi.mean() / 1.0)
            if epoch % 100 == 0 and int(os.environ['DEBUG']) == 1:
                print("Epoch: " + str(epoch) + ", policy loss: " + str(policy_losses[-1]))

        if int(os.environ['PLOT']) == 1:
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(list(range(len(policy_losses))), np.array(policy_losses), label="Policy loss")
            plt.xlim(left=0)
            plt.legend()
            # plt.ylim(bottom=0)
            plt.subplot(3, 1, 2)
            plt.plot(list(range(len(alphas))), np.array(alphas), label="alphas")
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.plot(list(range(len(log_pis))), np.array(log_pis), label="Entropy")
            plt.legend()
            plt.show(block=False)

        self.eval_statistics['policy_train_steps_total'] = self._n_train_steps_total
        self.end_epoch(epoch)

        return policy_losses[-1], self.get_diagnostics()

    def training_step(self, indices, step):
        # get data from replay buffer
        if step == 0:
            gt.stamp('pt_before_sample')
        batch_enc, enc_mask, batch_sac = \
            self.replay_buffer.sample_random_few_step_data_batch(indices, self.batch_size,
                                                                 normalize=self.use_data_normalization,
                                                                 normalize_sac=self.use_sac_data_normalization,
                                                                 return_sac_data=True)
        if step == 0:
            gt.stamp('pt_sample')

        rewards = ptu.from_numpy(batch_sac['rewards'])
        terminals = ptu.from_numpy(batch_sac['terminals'])
        obs = ptu.from_numpy(batch_sac['observations'])
        actions = ptu.from_numpy(batch_sac['actions'])
        next_obs = ptu.from_numpy(batch_sac['next_observations'])

        encoder_input, enc_mask = self.replay_buffer.make_encoder_data(batch_enc, self.batch_size, padding_mask=enc_mask)
        task_z, task_y = self.encoder(encoder_input, enc_mask)
        task_z = task_z.detach()
        task_y = task_y.detach()
        # Without rela
        # task_z = ptu.from_numpy(batch_sac['task_indicators'])
        # task_y = ptu.from_numpy(batch_sac['base_task_indicators'])
        # new_task_z = ptu.from_numpy(batch_sac['next_task_indicators'])
        # new_task_y = ptu.from_numpy(batch_sac['next_base_task_indicators'])
        if step == 0:
            gt.stamp('pt_to_torch')

        # for debug
        # task_z = torch.zeros_like(task_z)
        # new_task_z = torch.zeros_like(new_task_z)
        # task_z = torch.from_numpy(batch_sac['true_tasks'])
        # new_task_z = torch.cat([task_z[1:,:], task_z[-1,:].view(1,1)])

        # Todo: We overwrite the new task here. Is this really intentional?
        # It seems to work better, but why?
        new_task_z = task_z.clone().detach()
        new_task_y = task_y.clone().detach()

        # Variant 1: train the SAC as if there was no encoder and the state is just extended to be [state , z]
        # obs = torch.cat((obs, task_z), dim=1)
        # next_obs = torch.cat((next_obs, new_task_z), dim=1)

        """
        Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy_networks.forward(
            'policy',
            obs,
            task_z,
            task_y,
            reparameterize=True, return_log_prob=True,
        )
        if self.use_automatic_entropy_tuning:
            if self.use_parametrized_alpha:
                self.log_alpha = self.policy_networks.forward('alpha_net', None, task_z, task_y)
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.use_parametrized_alpha:
                alpha = self.log_alpha.exp().detach()
            else:
                alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self._alpha

        if step == 0:
            gt.stamp('pt_alpha')

        """
        Policy Loss
        """
        q_new_actions = torch.min(
            self.policy_networks.forward('qf1', obs, task_z, task_y, new_obs_actions),
            self.policy_networks.forward('qf2', obs, task_z, task_y, new_obs_actions),
        )
        if step == 0:
            gt.stamp('pt_q_forward')
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        QF Loss
        """
        q1_pred = self.policy_networks.forward('qf1', obs, task_z, task_y, actions)
        q2_pred = self.policy_networks.forward('qf2', obs, task_z, task_y, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self.policy_networks.forward(
            'policy',
            next_obs,
            new_task_z,
            new_task_y,
            reparameterize=True, return_log_prob=True
        )
        target_q_values = torch.min(
            self.policy_networks.forward('target_qf1', next_obs, new_task_z, new_task_y,  new_next_actions),
            self.policy_networks.forward('target_qf2', next_obs, new_task_z, new_task_y,  new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        if step == 0:
            gt.stamp('pt_q_target')

        """
        Update policy networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        if step == 0:
            gt.stamp('pt_policy_update')

        """
        Update QF networks
        """
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        if step == 0:
            gt.stamp('pt_q_update')

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            self.policy_networks.soft_target_update(self.soft_target_tau)

        if step == 0:
            gt.stamp('pt_q_softupdate')

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch_sac.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))
            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.mean().item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.mean().item()
        self._n_train_steps_total += 1

        if step == 0:
            gt.stamp('pt_statistics')

        return ptu.get_numpy(policy_loss), ptu.get_numpy(alpha), ptu.get_numpy(log_pi)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        d = self.policy_networks.get_networks()
        return [
            d['policy'],
            d['qf1'],
            d['qf2'],
            d['target_qf1'],
            d['target_qf2'],
        ]

    def get_snapshot(self):
        return self.policy_networks.get_networks()
