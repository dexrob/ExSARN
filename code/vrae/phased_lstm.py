import math

import torch
import torch.nn as nn

from datetime import datetime

class PhasedLSTMCell(nn.Module):
    """Phased LSTM recurrent network cell.
    https://arxiv.org/pdf/1610.09513v1.pdf
    """

    def __init__(
        self,
        hidden_size,
        leak=0.001,
        ratio_on=0.10, # 0.05 in the paper
        period_init_min=1.0,
        period_init_max=200.0 #  1000.0 in the paper
    ):
        """
        Args:
            hidden_size: int, The number of units in the Phased LSTM cell.
            leak: float or scalar float Tensor with value in [0, 1]. Leak applied
                during training.
            ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
                period during which the gates are open.
            period_init_min: float or scalar float Tensor. With value > 0.
                Minimum value of the initialized period.
                The period values are initialized by drawing from the distribution:
                e^U(log(period_init_min), log(period_init_max))
                Where U(.,.) is the uniform distribution.
            period_init_max: float or scalar float Tensor.
                With value > period_init_min. Maximum value of the initialized period.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.ratio_on = ratio_on
        self.leak = leak

        # initialize time-gating parameters
        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(
                math.log(period_init_min), math.log(period_init_max)
            )
        )
        self.tau = nn.Parameter(period)

        phase = torch.Tensor(hidden_size).uniform_() * period
        self.phase = nn.Parameter(phase)
        self.k = None

    def _compute_phi(self, t): # TODO
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)
        phase_ = self.phase.view(1, -1).repeat(t.shape[0], 1)
        tau_ = self.tau.view(1, -1).repeat(t.shape[0], 1)

        phi = torch.fmod((t_ - phase_), tau_).detach()
        phi = torch.abs(phi) / tau_
        return phi

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return x + (torch.fmod(x, y) - x).detach()

    def set_state(self, h, c):
        self.h0 = h
        self.c0 = c

    def forward(self, h_s, c_s, t):
        # print(c_s.size(), h_s.size(), t.size())
        phi = self._compute_phi(t)

        # Phase-related augmentations
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi

        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], t.shape[0], -1) # k: torch.Size([1, 32, 90])
        self.k = k

        h_s_new = k * h_s + (1 - k) * self.h0
        c_s_new = k * c_s + (1 - k) * self.c0

        return h_s_new, c_s_new

class PhasedLSTMCell_v2(nn.Module):
    """Phased LSTM recurrent network cell.
    https://arxiv.org/pdf/1610.09513v1.pdf
    same oscillation for each feature
    """

    def __init__(
        self,
        hidden_size,
        leak=0.001,
        ratio_on=0.10, # 0.05 in the paper
        period_init_min=1.0,
        period_init_max=200.0 #  1000.0 in the paper
    ):
        """
        Args:
            hidden_size: int, The number of units in the Phased LSTM cell.
            leak: float or scalar float Tensor with value in [0, 1]. Leak applied
                during training.
            ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of the
                period during which the gates are open.
            period_init_min: float or scalar float Tensor. With value > 0.
                Minimum value of the initialized period.
                The period values are initialized by drawing from the distribution:
                e^U(log(period_init_min), log(period_init_max))
                Where U(.,.) is the uniform distribution.
            period_init_max: float or scalar float Tensor.
                With value > period_init_min. Maximum value of the initialized period.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.ratio_on = ratio_on
        self.leak = leak

        # initialize time-gating parameters
        period = torch.exp(
            torch.Tensor(hidden_size).uniform_(
                math.log(period_init_min), math.log(period_init_max)
            )
        )
        self.tau = nn.Parameter(period)

        phase = torch.Tensor(hidden_size).uniform_() * period
        self.phase = nn.Parameter(phase)
        self.k = None

    def _compute_phi(self, t): # TODO
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)
        phase_ = self.phase.view(1, -1).repeat(t.shape[0], 1)
        tau_ = self.tau.view(1, -1).repeat(t.shape[0], 1)

        phi = torch.fmod((t_ - phase_), tau_).detach()
        phi = torch.abs(phi) / tau_
        return phi

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return x + (torch.fmod(x, y) - x).detach()

    def set_state(self, h, c):
        self.h0 = h
        self.c0 = c

    def forward(self, h_s, c_s, t):
        # print(c_s.size(), h_s.size(), t.size())
        phi = self._compute_phi(t)

        # Phase-related augmentations
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi

        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], t.shape[0], -1) # k: torch.Size([1, 32, 90])
        # print(k.size())
        k = torch.sum(k, dim=2, keepdim=True) / 90.0
        k = k.repeat(1, 1, self.hidden_size)
        self.k = k

        h_s_new = k * h_s + (1 - k) * self.h0
        c_s_new = k * c_s + (1 - k) * self.c0

        return h_s_new, c_s_new

class PhasedLSTM(nn.Module):
    """Wrapper for multi-layer sequence forwarding via
       PhasedLSTMCell"""

    def __init__(
        self,
        input_size,
        hidden_size,
        device, 
        batch_first=True,
        bidirectional=False, 
        ratio_on=0.001,
        period_init_max=1000.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device

        # self.lstm = nn.LSTM(
        #     input_size=input_size,
        #     hidden_size=hidden_size,
        #     bidirectional=bidirectional,
        #     batch_first=batch_first
        # )
        self.lstm = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

        self.bi = 2 if bidirectional else 1

        self.phased_cell = PhasedLSTMCell_v2(
            hidden_size=self.bi * hidden_size,
            ratio_on = ratio_on,
            period_init_max=period_init_max
        )

    def forward(self, u_sequence, times):
        """
        Args:
            sequence: The input sequence data of shape (batch, time, N)
            times: The timestamps corresponding to the data of shape (batch, time)
            `times` should be a matrix of shape
            ``(n_batch, n_time_steps)`` where ``times[i, j] = 1`` when ``j <
            (length of sequence i)`` and ``times[i, j] = 0`` when ``j >= (length
            of sequence i)`
        """
        startime = datetime.now()
        h0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size)).to(self.device)
        c0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size)).to(self.device)
        self.phased_cell.set_state(h0, c0)
        times = times.to(self.device)

        h_out = None
        c_out = None
        k_out = None
        for i in range(u_sequence.size(1)):
            # u_t = u_sequence[:, i, :].unsqueeze(1) # [32, 1, 90]
            # t_t = times[:, i]
            # _, (h_t, c_t) = self.lstm(u_t, (h0, c0))
            # (h_s, c_s) = self.phased_cell(h_t, c_t, t_t)
            u_t = u_sequence[:, i, :] # [32, 1, 90]
            t_t = times[:, i]
            h_t, c_t = self.lstm(u_t)
            h_t, c_t = h_t.unsqueeze(0), c_t.unsqueeze(0)
            (h_s, c_s) = self.phased_cell(h_t, c_t, t_t)
            k = self.phased_cell.k
            # print('h_t:', h_t.size()) # [1, 32, 90]
            # print('h_s:', h_s.size())

            self.phased_cell.set_state(h_s, c_s)
            c_0, h_0 = c_s.squeeze(), h_s.squeeze()

            if h_out is None:
                h_out = h_s
                c_out = c_s
                k_out = k
            else:
                # check dim
                if len(h_out.shape)==len(h_s.shape):
                    # 1st concatenation
                    h_out = torch.stack((h_out, h_s), 1)
                    c_out = torch.stack((c_out, c_s), 1)
                    k_out = torch.stack((k_out, k), 1)
                else:
                    h_out = torch.cat((h_out, h_s.unsqueeze(1)), 1)
                    c_out = torch.cat((c_out, c_s.unsqueeze(1)), 1)
                    k_out = torch.cat((k_out, k.unsqueeze(1)), 1)

        h_out = h_out.permute(0, 2, 1, 3)

        # print('forward time:', datetime.now()-startime)

        return h_out, (h_s, c_s), k_out