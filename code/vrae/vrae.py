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
from .phased_lstm import PhasedLSTM

class Encoder(nn.Module):
    # Encoder catered for HCNC
    """
    Encoder network containing enrolled LSTM
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: latent vector length
    :param dropout: percentage of nodes to dropout
    """
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, mlp_hidden=18, bidirectional=False, block="LSTM", device=None, ratio_on=None, period_init_max=None, bn=None):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.bidirectional = bidirectional
        self.bn = bn
        
        if mlp_hidden:
            self.use_mlp = True
            self.mlp_hidden = mlp_hidden
            self.fc = nn.Linear(self.number_of_features, self.mlp_hidden)
            if bn == True:
                # 
                self.bn_layer = nn.LayerNorm(self.mlp_hidden)
        
        else:
            self.use_mlp = False
            self.mlp_hidden = self.number_of_features
        
        if block == "LSTM":
            self.model = nn.LSTM(input_size=self.mlp_hidden, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional)
        elif block == "GRU":
            self.model = nn.GRU(input_size=self.mlp_hidden, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=self.bidirectional)
        elif block == "phased_LSTM":
            self.block = block
            self.model = PhasedLSTM(input_size=self.mlp_hidden,
                    hidden_size=self.hidden_size, 
                    batch_first=True,
                    bidirectional=self.bidirectional,
                    device=device,
                    ratio_on=ratio_on,
                    period_init_max=period_init_max)
        else:
            raise NotImplementedError("Unknown type for memory unit: {}".format(block))

    def forward(self, x, times=None):
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
                # if self.bn == True:
                #     out = self.bn_layer(out)
            else:
                out = x[...,t]
            embed_seq.append(out)
        embed_seq = torch.stack(embed_seq, dim=0).transpose_(0, 1)
        
        # for batch_first == True, input to lstm should be (batch, seq, feature)
        # forward on LSTM
        # self.lstm.flatten_parameters()
        if isinstance(self.model, nn.LSTM):
            r_out, (h_n, h_c) = self.model(embed_seq) 
        elif isinstance(self.model, nn.GRU):
            r_out, h_n = self.model(embed_seq)
        elif self.block == "phased_LSTM":
            r_out, (h_n, h_c), k_out = self.model(embed_seq, times) 
            self.k_out = k_out
        else:
            raise NotImplementedError("Unknown type for memory unit: {}".format(block))

        # cell_output = h_n[-1, :, :] = r_out[:,-1,:]
        # for batch_first = True,
        # output of shape (batch, seq_len,  num_directions * hidden_size):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)

        if self.bidirectional:
            # concat output from normal RNN and output from reverse RNN
            cell_output_bi = torch.cat((h_n[-1, :, :],h_n[0, :, :]), 1)
            print("cell_output_bi", cell_output_bi.shape)
            raise NotImplementedError("need to adjust lmbd layer for bidirectional rnn")
        else:
            cell_output = h_n[-1, :, :]
        
        # reshape to be consistent as decoder_output
        embed_seq = embed_seq.permute(0, 2, 1)
        return cell_output, r_out, embed_seq


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


class CnnEncoder(nn.Module):
    """
    Encoder network containing enrolled LSTM
    :param number_of_features: number of input features
    :param hidden_size: hidden size of the RNN
    :param hidden_layer_depth: number of layers in RNN
    :param latent_length: length of the latent vector
    :param dropout: percentage of nodes to dropout
    """
    def __init__(self, number_of_features, block, hidden_size, hidden_layer_depth, latent_length, dropout, cnn_number_of_features=None, device=None, ratio_on=None, period_init_max=None):

        super(CnnEncoder, self).__init__()
        # overwrite number_of_features, since data -> CNN -> LSTM
        if cnn_number_of_features is not None:
            self.number_of_features = cnn_number_of_features
        else:
            self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.cnn = CNN(cnn_number_of_features=cnn_number_of_features)
        if block == 'LSTM':
            self.model = nn.LSTM(input_size=self.number_of_features, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,dropout=dropout)
        elif block == "GRU":
            self.model = nn.GRU(input_size=self.number_of_features, 
                    hidden_size=self.hidden_size, 
                    num_layers=self.hidden_layer_depth, 
                    batch_first=True,
                    dropout=dropout)
        elif block == "phased_LSTM":
            self.block = block
            self.model = PhasedLSTM(input_size=self.number_of_features,
                    hidden_size=self.hidden_size, 
                    batch_first=True,
                    device=device,
                    ratio_on=ratio_on,
                    period_init_max=period_init_max)
        else:
            raise NotImplementedError("Unknown type for memory unit: {}".format(block))

    def forward(self, x, times=None):
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

        # forward on LSTM
        if isinstance(self.model, nn.LSTM):
            self.model.flatten_parameters()
            r_out, (h_n, c_n) = self.model(cnn_embed_seq)
        elif isinstance(self.model, nn.GRU):
            r_out, h_n = self.model(cnn_embed_seq)
        elif self.block == "phased_LSTM":
            r_out, (h_n, h_c), k_out = self.model(cnn_embed_seq, times) 
            self.k_out = k_out
        else:
            raise NotImplementedError("Unknown type for memory unit: {}".format(block))

        h_end = h_n[-1, :, :]

        return h_end, mp_indices


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
    def __init__(self, num_class, block, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
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

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        else:
            raise NotImplementedError
        
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
        
        if self.header is None:
            self.decoder = Decoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        elif self.header == "CNN": 
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        
        else:
            raise NotImplementedError
            
        
        self.sequence_length = sequence_length
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

    def forward(self, x, times=None):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output, r_out, embed_seq = self.encoder(x, times) 
            latent = self.lmbd(cell_output)
            # x_decoded 
            self.latent = latent
            self.cell_output = cell_output
            _, x_decoded, _= self.decoder(latent, times)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x, times)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices, times)
        else:
            raise NotImplementedError
        output = self.classifier(latent)

        return x_decoded, latent, output
   
class VRAEC_v2(BaseEstimator, nn.Module):
    # for properties discrimination
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
    def __init__(self, num_class_texture, block, sequence_length, number_of_features, 
                 num_class_stiffness=2, stiffness_latent_length=10, 
                 num_class_roughness=2, roughness_latent_length=10,
                 hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, n_epochs=5, dropout_rate=0., cuda=False,
                 dload='.', model_name='model', header=None, device='cpu', var=False, ratio_on=None, period_init_max=None):
        # RH: add mlp_hidden as the preprocessing unit

        super(VRAEC_v2, self).__init__()

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

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        else:
            raise NotImplementedError
        
        var = int(var) # note: default is string, 0/1 doesn't help
        
        if var:
            self.lmbd = VarLambda(hidden_size=hidden_size,
                                  latent_length=latent_length)
        else:
            self.lmbd = Lambda(hidden_size=hidden_size,
                               latent_length=latent_length)

        self.classifier_texture = nn.Sequential(
            nn.Linear(latent_length, num_class_texture),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1)
        )

        self.classifier_stiffness = nn.Sequential(
            Lambda(hidden_size=latent_length, latent_length=stiffness_latent_length),
            nn.Linear(stiffness_latent_length, num_class_stiffness),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1))

        self.classifier_roughness = nn.Sequential(
            Lambda(hidden_size=latent_length, latent_length=roughness_latent_length),
            nn.Linear(roughness_latent_length, num_class_roughness),
            # nn.Dropout(0.2),
            nn.LogSoftmax(dim=1))
        
        if self.header is None:
            self.decoder = Decoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        elif self.header == "CNN": 
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        
        else:
            raise NotImplementedError
            
        
        self.sequence_length = sequence_length
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

    def forward(self, x, times=None):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output, r_out, embed_seq = self.encoder(x, times) 
            latent = self.lmbd(cell_output)
            # x_decoded 
            self.latent = latent
            self.cell_output = cell_output
            _, x_decoded, _= self.decoder(latent, times)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x, times)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices, times)
        else:
            raise NotImplementedError
        output_texture = self.classifier_texture(latent)

        output_stiffness = self.classifier_stiffness(latent)
        output_roughness = self.classifier_roughness(latent)

        return x_decoded, latent, output_texture, output_stiffness, output_roughness

   
class VRAEC_bn(BaseEstimator, nn.Module):
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
    def __init__(self, num_class, block, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, n_epochs=5, dropout_rate=0., cuda=False,
                 dload='.', model_name='model', header=None, device='cpu', var=False, ratio_on=None, period_init_max=None):
        # RH: add mlp_hidden as the preprocessing unit

        super(VRAEC_bn, self).__init__()

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

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max,
                                bn=False)

        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max,
                                bn=True)

        else:
            raise NotImplementedError
        
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
        
        if self.header is None:
            self.decoder = Decoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        elif self.header == "CNN": 
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        
        else:
            raise NotImplementedError
            
        
        self.sequence_length = sequence_length
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

        self.ln2 = nn.LayerNorm(latent_length)
        self.ln1 = nn.LayerNorm(hidden_size)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x, times=None):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output, r_out, embed_seq = self.encoder(x, times) 
            cell_output = self.ln1(cell_output)
            latent = self.lmbd(cell_output)
            # x_decoded 

            self.latent = latent
            self.cell_output = cell_output
            _, x_decoded, _= self.decoder(latent, times)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x, times)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices, times)
        else:
            raise NotImplementedError
        latent = self.ln2(latent)
        output = self.classifier(latent)

        return x_decoded, latent, output
   
    
class VRAEC_property_layer(BaseEstimator, nn.Module):
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
    def __init__(self, num_class, block, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, n_epochs=5, dropout_rate=0., cuda=False,
                 dload='.', model_name='model', header=None, device='cpu', var=False, ratio_on=None, period_init_max=None):
        # RH: add mlp_hidden as the preprocessing unit

        super(VRAEC_property_layer, self).__init__()

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

        if self.header is None:
            self.encoder = Encoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        elif self.header == "CNN":
            self.encoder = CnnEncoder(number_of_features = number_of_features,
                                block=block,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                dropout=dropout_rate,
                                cnn_number_of_features=18,
                                device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)

        else:
            raise NotImplementedError
        
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
        
        if self.header is None:
            self.decoder = Decoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype, device=self.device,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        elif self.header == "CNN": 
            self.decoder = CnnDecoder(sequence_length=sequence_length,
                                block=block,
                                batch_size = batch_size,
                                hidden_size=hidden_size,
                                hidden_layer_depth=hidden_layer_depth,
                                latent_length=latent_length,
                                output_size=number_of_features,
                                dtype=self.dtype,
                                device=self.device,
                                cnn_number_of_features=18,
                                ratio_on=ratio_on,
                                period_init_max=period_init_max)
        
        else:
            raise NotImplementedError
            
        
        self.sequence_length = sequence_length
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
        self.stiff_layer = nn.Linear(1, 1)
        self.rough_layer = nn.Linear(1, 1)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x, times=None):
        """
        Forward propagation which involves one pass from inputs to encoder to lambda to decoder
        :param x:input tensor
        :return: the decoded output, latent vector
        """
        if self.header is None:
            cell_output, r_out, embed_seq = self.encoder(x, times) 
            latent = self.lmbd(cell_output)
            # x_decoded 
            self.latent = latent
            self.cell_output = cell_output
            _, x_decoded, _= self.decoder(latent, times)
        elif self.header == "CNN":
            cell_output, mp_indices = self.encoder(x, times)
            latent = self.lmbd(cell_output)
            x_decoded = self.decoder(latent, mp_indices, times)
        else:
            raise NotImplementedError
        output = self.classifier(latent)
        stiff_latent = latent[:, 0].reshape(-1, 1)
        rough_latent = latent[:, 1].reshape(-1, 1)
        s = self.stiff_layer(stiff_latent)
        s = torch.squeeze(s)
        r = self.rough_layer(rough_latent)
        r = torch.squeeze(r)

        return x_decoded, latent, output, s, r
  