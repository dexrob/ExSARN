''' Code is modified based on https://github.com/tejaslodaya/timeseries-clustering-vae'''
import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import os, sys
import pickle
# from .phased_lstm import PhasedLSTM
import seaborn as sns

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
        period_init_max=400.0 #  1000.0 in the paper
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
        self.tau1 = nn.Parameter(period)
        self.tau2 = nn.Parameter(period / 400 * 75)

        phase = torch.Tensor(hidden_size).uniform_() * period
        self.phase1 = nn.Parameter(phase)
        self.phase2 = nn.Parameter(phase / 400 * 75 + (400-75))
        self.k = None

    def _compute_phi(self, t): # TODO
        t_ = t.view(-1, 1).repeat(1, self.hidden_size)
        phase_1 = self.phase1.view(1, -1).repeat(t.shape[0], 1)
        tau_1 = self.tau1.view(1, -1).repeat(t.shape[0], 1)

        phi1 = torch.fmod((t_ - phase_1), tau_1).detach()
        phi1 = torch.abs(phi1) / tau_1

        phase_2 = self.phase2.view(1, -1).repeat(t.shape[0], 1)
        tau_2 = self.tau2.view(1, -1).repeat(t.shape[0], 1)

        phi2 = torch.fmod((t_ - phase_2), tau_2).detach()
        phi2 = torch.abs(phi2) / tau_2
        phi = torch.cat((phi1, phi2))
        # sns_plot = sns.heatmap(phi.detach().numpy())
        # sns_plot.get_figure().savefig('plot.png')
        # sns_plot.get_figure().clf()
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
        # print(t)

        # Phase-related augmentations
        k_up = 2 * phi / self.ratio_on
        k_down = 2 - k_up
        k_closed = self.leak * phi
        
        k = torch.where(phi < self.ratio_on, k_down, k_closed)
        k = torch.where(phi < 0.5 * self.ratio_on, k_up, k)
        k = k.view(c_s.shape[0], t.shape[0], -1) 
        k = torch.sum(k, dim=2, keepdim=True) / k.size(2)
        k = k.repeat(1, 1, self.hidden_size)
        if t[0] < 325: # 400-75
            k[:, 32:, :] = 0
        self.k = k
        # print(k.size()) # [1, 64, 90]
        # print(self.h0.size()) # [1, 64, 90]
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
        ratio_on=0.10,
        period_init_max=400.0 # use the larger number of time step
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
        # print(u_sequence.size())
        h0 = u_sequence.new_zeros((self.bi, u_sequence.size(0), self.hidden_size)).to(self.device)
        # print(h0.size())
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

class B_header(nn.Module):
    # Encoder catered for HCNC
    """
    Encoder network containing enrolled LSTM
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    """
    def __init__(self, number_of_features, mlp_hidden=18, device=None):

        super(B_header, self).__init__()

        self.number_of_features = number_of_features
        
        if mlp_hidden:
            self.use_mlp = True
            self.mlp_hidden = mlp_hidden
            self.fc = nn.Linear(self.number_of_features, self.mlp_hidden)
        
        else:
            self.use_mlp = False
            self.mlp_hidden = self.number_of_features

    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (batch_size, number_of_features, sequence_length)
        :return: last hidden state of encoder, of shape (batch_size, hidden_size)
        lstm: h_n of shape (num_layers, batch, hidden_size)
        max_timesteps: the length of the longest time sequence
        
        """
        batch_size, num_features, sequence_size = x.size()
        
        # create embedding
        embed_seq = []
        for t in range(sequence_size):
            if self.use_mlp:
                out = self.fc(x[...,t])
            else:
                out = x[...,t]
            embed_seq.append(out)
        embed_seq = torch.stack(embed_seq, dim=0).transpose_(0, 1)
        # print(embed_seq.size()) # [32, 400, 18]

        return embed_seq



class CNN(nn.Module):
    """
    CNN header network for iCub sensor
    :param C: number of channels of the taxel image
    :param H: height of the taxel image
    :param W: width of the taxel image
    :param cnn_number_of_features: number of CNN output features, also equivalent to number of input featurs to CNNEncoder
    """
    def __init__(self, C=1, H=6, W=10, cnn_number_of_features=18): # todo
        super(CNN, self).__init__()
        self.cnn_number_of_features = cnn_number_of_features
        self.C = C
        self.H = H 
        self.W = W

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=self.C, out_channels=3, kernel_size=(3, 5)),
            nn.MaxPool2d(2, return_indices=True),
        )

    def forward(self, x):
        """
        Forward propagation of CNN. Given input, outputs the CNN feature and mapping indices
        
        :param x: input to the CNN, of shape (batch_size, channel, height, width)
        :return cnn_out: cnn output feature
        :return mp_indices: mapping indices for convolution
        
        """
        # check input size 
        batch_size, C, H, W = x.size()
        assert C==self.C and H==self.H and W==self.W, "wrong size for CNN input, x {}, \
            should be (batch_size,{},{},{})".format(x.size(), self.C, self.H, self.W)
        cnn_out, mp_indices = self.seq(x)
        cnn_out = cnn_out.view(-1, self.cnn_number_of_features)
        return cnn_out, mp_indices


class I_header(nn.Module):
    """
    Encoder network containing enrolled LSTM
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param dropout: percentage of nodes to dropout
    """
    def __init__(self, number_of_features, cnn_number_of_features=None, device=None):

        super(I_header, self).__init__()
        # overwrite number_of_features, since data -> CNN -> LSTM
        self.device = device
        if cnn_number_of_features is not None:
            self.number_of_features = cnn_number_of_features
        else:
            self.number_of_features = number_of_features
        
        self.cnn = CNN(cnn_number_of_features=cnn_number_of_features)
        
    def forward(self, x):
        """
        Forward propagation of encoder. Given input, outputs the last hidden state of encoder
        
        :param x: input to the encoder, of shape (sequence_length, batch_size, H, W, sequence_size)
        :return h_end: last hidden state of encoder, of shape (batch_size, hidden_size)
        :return mp_indices: keep mapping indices for reshaping later in decoder
        lstm: h_n of shape (num_layers, batch, hidden_size)
        
        """
        x = x.unsqueeze(1)
        batch_size, C, H, W, sequence_size = x.size()
        
        # create CNN embedding
        cnn_embed_seq = []
        for t in range(sequence_size):
            cnn_out, mp_indices = self.cnn(x[...,t])
            cnn_embed_seq.append(cnn_out)   
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1) # 32, 75, 18
        zero_pad = torch.ones(cnn_embed_seq.size(0), 400-75, cnn_embed_seq.size(2)).to(self.device)
        cnn_embed_seq = torch.cat((zero_pad, cnn_embed_seq), 1) # 32, 400, 18

        return cnn_embed_seq, mp_indices


class Lambda(nn.Module):
    """
    Lambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: length of the latent vector
    """
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()
 
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        # RH: pay attention to the init
        nn.init.xavier_uniform_(self.hidden_to_mean.weight)

    def forward(self, cell_output):
        """
        Given last hidden state of encoder, passes through a linear layer, and finds its mean value
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """
        
        return self.hidden_to_mean(cell_output)
            

class VarLambda(nn.Module):
    """VarLambda module converts output of encoder to latent vector
    :param hidden_size: hidden size of the encoder
    :param latent_length: latent vector length
    """
    def __init__(self, hidden_size, latent_length):
        super(VarLambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        """Given last hidden state of encoder, passes through a linear layer, and finds the mean and variance
        :param cell_output: last hidden state of encoder
        :return: latent vector
        """

        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)
        
        # RH: dunno where to set self.training, but it indeed enters "if" clause
        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean    
        
       
class Decoder(nn.Module):
    # Decoder catered for HCNC
    """
    Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param output_size: output size of the mean vector
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    :param device: Depending on cuda enabled/disabled
    """
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device, mlp_hidden=None, bidirectional=False, block="LSTM", ratio_on=None, period_init_max=None):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype
        self.device = device
        self.bidirectional = bidirectional
        
        if mlp_hidden is None:
            self.use_mlp = False
            self.mlp_hidden = output_size
        else:
            self.use_mlp = True
            self.mlp_hidden = mlp_hidden
            self.dfc = nn.Linear(self.mlp_hidden, output_size)
            # mlp_hidden eqivalent to num_features
        
        if self.bidirectional:
            self.ndirection = 2
        else:
            self.ndirection = 1
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        if block == "LSTM":
            self.model = nn.LSTM(input_size=self.hidden_size, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True)
        elif block == "GRU":
            self.model = nn.GRU(input_size=self.hidden_size, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,)
        elif block == "phased_LSTM":
            self.block = block
            self.model = PhasedLSTM(input_size=self.hidden_size,
                    hidden_size=self.hidden_size, 
                    batch_first=True,
                    device=device,
                    ratio_on=ratio_on,
                    period_init_max=period_init_max)
        
        self.hidden_to_output = nn.Linear(self.ndirection*self.hidden_size, self.mlp_hidden) 
        
        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, times=None):
        """
        Converts latent to hidden to output
        :param latent: latent vector
        :return: output consisting of mean vector
        """
        
        latent_input = self.latent_to_hidden(latent)
        decoder_input = latent_input.repeat(self.sequence_length, 1, 1).transpose_(0, 1) # [8, 400, 90]
        if isinstance(self.model, nn.LSTM):
            decoder_output, (h_n, c_n) = self.model(decoder_input)
        elif isinstance(self.model, nn.GRU):
            decoder_output, h_n = self.model(decoder_input)
        elif self.block == "phased_LSTM":
            decoder_output, (h_n, c_n), k_out = self.model(decoder_input, times) 
            self.k_out = k_out
        else:
            raise NotImplementedError("Unknown type for memory unit: {}".format(block))
        out = self.hidden_to_output(decoder_output) # [400, 8, 19] 
        out = torch.squeeze(out)
        batch_size, sequence_size, number_of_features = out.size()
        out = out.permute(0, 2, 1)

        # mirror fc embedding
        dfc_seq =[]
        self.hidden_seq = []
        for t in range(sequence_size):
            x = out[..., t].view(batch_size,-1)
            # print("one stamp of out size", x.size()) #[32, 6]
            if self.use_mlp:
                self.hidden_seq.append(x) # save hidden representation
                x = self.dfc(x)
            # print("new x size", x.size()) #[32, 19]
            dfc_seq.append(x)
        if len(self.hidden_seq):
            self.hidden_seq = torch.stack(self.hidden_seq, dim=0).transpose_(0,1)
            self.hidden_seq = self.hidden_seq.reshape(batch_size, -1, sequence_size)
        dfc_seq = torch.stack(dfc_seq, dim=0).transpose_(0, 1)
        dfc_seq = dfc_seq.reshape(batch_size, -1, sequence_size) #[32, 19, 400]
        return dfc_seq, out, latent_input

    
#     def init_hidden(self, batch_size):
#         # Modify on 9.28 to cater to changing batch_size, not sure whether it will affect performance
#         self.decoder_inputs = torch.zeros(self.sequence_length, batch_size, 1, requires_grad=True).to(self.device)
#         self.c_0 = torch.zeros(self.hidden_layer_depth, batch_size, self.hidden_size, requires_grad=True).to(self.device)
        


class CnnDecoder(nn.Module):
    """
    Converts latent vector into output
    :param sequence_length: length of the input sequence
    :param batch_size: batch size of the input sequence
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param output_size: output size 
    :param dtype: Depending on cuda enabled/disabled, create the tensor
    :param device: Depending on cuda enabled/disabled
    :param cnn_number_of_feratures: number of features of cnn output, equivalent to lstm input
    """
    def __init__(self, sequence_length, block, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, device, cnn_number_of_features=None, ratio_on=None, period_init_max=None):

        super(CnnDecoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        # manually specify the C, H, W, consistent with CNNEncoder
        self.C = 1
        self.H = 6
        self.W = 10

        if cnn_number_of_features is None:
            self.output_size = output_size
        else: 
            self.output_size = cnn_number_of_features
        self.device = device

        # mirror CNN
        self.unpool = nn.MaxUnpool2d(2)
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size=(3, 5)),
            nn.ReLU()
        )
        if block == "LSTM":
            self.model = nn.LSTM(self.hidden_size, self.hidden_size, self.hidden_layer_depth)
        elif block == "GRU":
            self.model = nn.GRU(self.hidden_size, self.hidden_size, self.hidden_layer_depth)
        elif block == "phased_LSTM":
            self.block = block
            self.model = PhasedLSTM(self.hidden_size, self.hidden_size, batch_first=True, device=device, ratio_on=ratio_on,
                    period_init_max=period_init_max)
        
        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent, mp_indices, times=None):
        """
        Converts latent to hidden to output
        :param latent: latent vector, mp_indices to reverse maxpooling correctly
        :param mp_indices: mapping indices for reshaping in decoder 
        :return: output consisting of mean vector
        """
        
        h_state = self.latent_to_hidden(latent)
        decoder_input = h_state.repeat(self.sequence_length, 1, 1).transpose_(0, 1) # [8, 400, 90]
        
        if isinstance(self.model, nn.LSTM):
            decoder_output, _ = self.model(decoder_input)
        elif isinstance(self.model, nn.GRU):
            decoder_output, _ = self.model(decoder_input)
        elif self.block == "phased_LSTM":
            decoder_output, (h_n, c_n), k_out = self.model(decoder_input, times) 
            decoder_output = torch.squeeze(decoder_output)
            self.k_out = k_out
        else:
            raise NotImplementedError
        
        out = self.hidden_to_output(decoder_output)
        # print("in cnn decoder: ", out.size()) # [75, 32, 18]
        batch_size, sequence_size, number_of_features = out.size()
        out = out.permute(0, 2, 1)
        # mirror CNN embedding
        dcnn_seq = []
        for t in range(sequence_size):
            x = out[..., t].view(batch_size,3,2,3)
            x = self.unpool(x, mp_indices) 
            x = self.dcnn(x)
            dcnn_seq.append(x)
            
        dcnn_seq = torch.stack(dcnn_seq, dim=0).transpose_(0, 1)
    
        dcnn_seq = dcnn_seq.reshape(batch_size, self.H, self.W, sequence_size)

        return dcnn_seq

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

    
class VRAEC(BaseEstimator, nn.Module):
    """
    Variational recurrent auto-encoder with classifier.
    This module is used for dimensionality reduction of timeseries and perform classification using hidden representation.
    :param num_class: number of class labels
    :param sequence_length: length of the input sequence
    :param number_of_features: number of input features
    :param hidden_size:  hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param batch_size: number of timeseries in a single batch
    :param learning_rate: the learning rate of the module
    :param n_epochs: Number of iterations/epochs
    :param dropout_rate: The probability of a node being dropped-out
    :param boolean cuda: to be run on GPU or not
    :param dload: Download directory where models are to be dumped. Currently saving model outside.
    :param model_name name of state dict to be stored under dload directory, without post
    :param header: "CNN" or "None", hearder implemented before encoder and after decoder
    """
    def __init__(self, num_class, block, I_sequence_length, I_number_of_features, B_sequence_length, B_number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, n_epochs=5, dropout_rate=0., cuda=False,
                 dload='.', model_name='model', header=None, device='cpu', var=False, ratio_on=None, period_init_max=None):
        # RH: add mlp_hidden as the preprocessing unit

        super(VRAEC, self).__init__()

        self.dtype = torch.FloatTensor
        self.ydtype = torch.LongTensor
        self.use_cuda = cuda
        self.header = header
        self.device = device
        self.epoch_train_acc = []
        
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor
            self.ydtype = torch.cuda.LongTensor

        self.header1 = B_header(number_of_features = B_number_of_features,
                                device=self.device)

        self.header2 = I_header(I_number_of_features,
                                cnn_number_of_features=18, device=self.device)

        if block == "phased_LSTM":
            self.block = block
            self.enc_rnn = PhasedLSTM(input_size=18,
                    hidden_size=hidden_size, 
                    batch_first=True,
                    device=device,
                    ratio_on=ratio_on,
                    period_init_max=period_init_max)
            self.dec_rnn = PhasedLSTM(input_size=hidden_size,
                    hidden_size=18, 
                    batch_first=True,
                    device=device,
                    ratio_on=ratio_on,
                    period_init_max=period_init_max)
        
        var = int(var) # note: default is string, 0/1 doesn't help
        
        if var:
            self.lmbd = VarLambda(hidden_size=hidden_size,
                                  latent_length=latent_length)
        else:
            self.lmbd = Lambda(hidden_size=hidden_size,
                               latent_length=latent_length)

        self.classifier = nn.Sequential(
            nn.Linear(latent_length, num_class),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )
        
        self.decoder1 = Decoder(sequence_length=B_sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=B_number_of_features,
                                dtype=self.dtype, device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        self.decoder2 = CnnDecoder(sequence_length=I_sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=I_number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        
        # self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.is_fitted = False
        self.dload = dload
        self.model_name = model_name

        self.latent = None
        self.cell_output = None

        if self.use_cuda:
            self.cuda()

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, XB, XI, times=None):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        # header 1
        embed_seq = self.header1(XB) 
        # header 2
        cnn_embed_seq, mp_indices = self.header2(XI)
        # x_decoded 
       #  _, x_decoded1, _= self.decoder1(cell_output1, times)
        # x_decoded2 = self.decoder2(cell_output2, mp_indices, times)

        cell_output = torch.cat((embed_seq, cnn_embed_seq), 0)
        # TODO run
        r_out, (h_n, h_c), k_out = self.enc_rnn(cell_output, times) 
        self.k_out = k_out
        # decoder_output, (h_n, h_c), k_out = self.dec_rnn(embed_seq, times) 
        # self.k_out = k_out
        x_decoded1 = None
        x_decoded2 = None

        latent = self.lmbd(h_n)
        self.latent = latent 

        output = self.classifier(latent)
        output = torch.squeeze(output)

        return x_decoded1, x_decoded2, latent, output
   