import torch
import torch.nn as nn
# TODO: Check if we can get rid of the reference to rlkit and implement MLP ourselves
from rlkit.torch.networks import Mlp
from abc import ABC, abstractmethod


class SharedEncoderBase(nn.Module, ABC):
    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec
    ):
        super(SharedEncoderBase, self).__init__()
        self.encoder_input_dim = state_dim + acton_dim + reward_dim + state_dim
        self.shared_dim = int(net_complex_enc_dec * self.encoder_input_dim)

    @abstractmethod
    def forward(self, x):
        pass

    @classmethod
    @property
    @abstractmethod
    def returns_timestep_encodings(cls):
        pass


class SharedEncoderTimestepMLP(SharedEncoderBase):
    returns_timestep_encodings = True

    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec
    ):
        super(SharedEncoderTimestepMLP, self).__init__(state_dim, acton_dim, reward_dim, net_complex_enc_dec)
        self.layers = Mlp(input_size=self.encoder_input_dim,
                          hidden_sizes=[self.shared_dim, self.shared_dim],
                          output_size=self.shared_dim,
                          )

    def forward(self, x):
        return self.layers(x)


class SharedEncoderTrajectoryMLP(SharedEncoderBase):
    returns_timestep_encodings = False

    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec,
            time_steps
    ):
        super(SharedEncoderTrajectoryMLP, self).__init__(state_dim, acton_dim, reward_dim, net_complex_enc_dec)

        self.time_steps = time_steps
        self.encoder_input_dim = time_steps * self.encoder_input_dim

        self.layers = Mlp(input_size=self.encoder_input_dim,
                          hidden_sizes=[self.shared_dim, self.shared_dim],
                          output_size=self.shared_dim,
                          )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layers(x)


class SharedEncoderGRU(SharedEncoderBase):
    returns_timestep_encodings = False

    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec
    ):
        super(SharedEncoderGRU, self).__init__(state_dim, acton_dim, reward_dim, net_complex_enc_dec)

        self.layers = nn.GRU(
            input_size=self.encoder_input_dim,
            hidden_size=self.shared_dim,
            batch_first=True
        )

    def forward(self, x):
        _, h = self.layers(x)
        return h.squeeze(dim=0)


class SharedEncoderConv(SharedEncoderBase):
    returns_timestep_encodings = False

    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec,
    ):
        super(SharedEncoderConv, self).__init__(state_dim, acton_dim, reward_dim, net_complex_enc_dec)

        self.layers = nn.Sequential(
            nn.Conv1d(self.encoder_input_dim, self.shared_dim, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Conv1d(self.shared_dim, self.shared_dim, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Flatten(start_dim=1),
            nn.LazyLinear(out_features=self.shared_dim)
        )

    def forward(self, x):
        # Input is in format (batch, time_step, feature), but Conv1d expects (batch, feature, time_step)
        x = x.permute((0, 2, 1))
        return self.layers(x)


class SharedEncoderFCN(SharedEncoderBase):
    returns_timestep_encodings = False

    def __init__(
            self,
            state_dim,
            acton_dim,
            reward_dim,
            net_complex_enc_dec,
    ):
        super(SharedEncoderFCN, self).__init__(state_dim, acton_dim, reward_dim, net_complex_enc_dec)

        self.layers = nn.Sequential(
            nn.Conv1d(self.encoder_input_dim, self.shared_dim, kernel_size=5),
            nn.ReLU(),
            nn.Conv1d(self.shared_dim, self.shared_dim, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(output_size=1)
        )

    def forward(self, x):
        # Input is in format (batch, time_step, feature), but Conv1d expects (batch, feature, time_step)
        x = x.permute((0, 2, 1))
        x = self.layers(x)
        return x.squeeze(dim=2)
