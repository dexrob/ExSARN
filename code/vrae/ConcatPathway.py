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
import math
# from vrae.lmu import LMU

class Receptor(nn.Module):
    """
    Emulate a single neuron, which receives a sequenctial signal and fires at certain frequency
    Receptor is similar to NestedEncoder in nested_vrae.py but uses a single lstm
    Everytime after it fires, it is reinitialized (probably with some residual)
    """
    def __init__(self, receptor_freq, receptor_input_size, receptor_hidden_size, receptor_hidden_layer_depth=1, dropout=0, bidirectional=False, block="LSTM", residual=0, device="cpu", norm=None):
        super(Receptor, self).__init__()
        
        self.norm = norm
        if self.norm is None:
            print("No normalization for receptor")
        elif self.norm == "BN":
            print("use BN for receptor, num_feature  {}".format(receptor_hidden_size))
            self.bn = nn.BatchNorm1d(num_features=receptor_hidden_size)
        else:
            raise NotImplementedError("unknown normalization for MLP eheader")
        
        self.receptor_freq = receptor_freq
        self.receptor_input_size = receptor_input_size
        self.block = block
        self.residual = residual
               
        if bidirectional:
            raise NotImplementedError("Have not implemented bidirection yet")
        print("{} block for receptors".format(self.block))
        if self.block == "LSTM":
            self.model = nn.LSTM(input_size = self.receptor_input_size,
                                hidden_size = receptor_hidden_size,
                                num_layers = receptor_hidden_layer_depth,
                                batch_first = True,
                                dropout = dropout,
                                bidirectional = bidirectional)
        elif self.block == "GRU":
            self.model = nn.GRU(input_size = self.receptor_input_size,
                               hidden_size = receptor_hidden_size,
                               num_layers = receptor_hidden_layer_depth,
                               batch_first = True,
                               dropout = dropout,
                               bidirectional = bidirectional)
        else:
            raise NotImplementedError("block {} if not implemented for receptor yet".format(block))
        
        self.device = device
        
    def forward(self, x, x_freq):
        # check the input shape
#         print("enter receptor, x shape", x.shape)
        # for batch_first = True, input to lstm should be (batch, seq, feature)
        # h_n (num_layers * num_directions, batch, hidden_size)
        batch_size, seq_len, num_feature = x.shape
        # initialize the hidden state
        h_0 = None
        c_0 = None
        assert num_feature == self.receptor_input_size, "uncompatible input size {} with model input {}".format(num_feature, self.receptor_input_size)
        sampling_period = x_freq/self.receptor_freq
        h_n_len = seq_len/x_freq*self.receptor_freq
        h_n_stack = []
        while len(h_n_stack)<h_n_len:
            # sample once
            start = math.floor(len(h_n_stack)*sampling_period)
            end = math.floor((len(h_n_stack)+1)*sampling_period)
            if h_0 is None:
                if self.block == "LSTM":
                    r_out, (h_n, c_n) = self.model(x[:,start:end,:])
                    # TODO: trick to initialize the neuron after refraction
                    h_0 = h_n*self.residual
                    c_0 = c_n*self.residual
                elif self.block == "GRU":
                    r_out, h_n = self.model(x[:,start:end,:])
                    h_0 = h_n*self.residual
                else:
                    raise NotImplementedError("block {} if not implemented for receptor yet".format(block))
                    
            else:
                if self.block == "LSTM":
                    r_out, (h_n, c_n) = self.model(x[:,start:end,:],(h_0, c_0))
                    h_0 = h_n*self.residual
                    c_0 = c_n*self.residual
                elif self.block == "GRU":
                    r_out, h_n = self.model(x[:,start:end,:], h_0)
                    h_0 = h_n*self.residual
                else:
                    raise NotImplementedError("block {} if not implemented for receptor yet".format(block))
                    
            h_n_stack.append(h_n)
        h_n_stack = torch.cat(h_n_stack, 0)
        # permute to batch first
        h_n_stack = h_n_stack.permute(1, 0, 2)
        if self.norm=="BN":
            batch, seq_len, feature = h_n_stack.shape
            h_n_stack = torch.reshape(h_n_stack, (-1, feature))
            h_n_stack = self.bn(h_n_stack)
            h_n_stack = torch.reshape(h_n_stack,(batch, seq_len, feature))
        return h_n_stack

class MLP(nn.Module):
    """
    MLP header for BioTac dataset
    """
    def __init__(self, input_size, output_size, dropout=0, norm=None):
        
        super(MLP, self).__init__()
        self.norm = norm
        if self.norm is None:
            print("No normalizaiton for MLP eheader")
        elif self.norm == "BN":
            print("use BN for MLP eheader")
            self.bn = nn.BatchNorm1d(num_features=output_size)
        else:
            raise NotImplementedError("unknown normalization for MLP eheader")
            
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.linear(x)
        if self.norm=="BN":
            batch, seq_len, feature = x.shape
            x = torch.reshape(x, (-1, feature))
            x = self.bn(x)
            x = torch.reshape(x,(batch, seq_len, feature))
        return self.dropout(self.activation(x))

    
class CNN(nn.Module):
    """
    CNN header for RoboSkin dataset
    """
    def __init__(self,C=1, H=6, W=10, output_size=18, norm=None):
        super(CNN, self).__init__()
        self.norm = norm
        
        if self.norm is None:
            print("No normalizaiton for CNN eheader")
        elif self.norm == "BN":
            print("use BN for CNN eheader")
        else:
            raise NotImplementedError("unknown normalization for CNN eheader")
        
        self.output_size = output_size
        self.C = C
        self.H = H
        self.W = W
        
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.C, out_channels=3, kernel_size=(3, 5)),
            nn.MaxPool2d(2, return_indices=True))
        
        self.bn = nn.BatchNorm1d(num_features=output_size)
    
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
        cnn_out, mp_indices = self.model(x)
        cnn_out = cnn_out.view(-1, self.output_size)
        if self.norm=="BN":
            cnn_out = self.bn(cnn_out)
        return cnn_out, mp_indices
    


class Latent(nn.Module):
    def __init__(self, input_size, output_size, norm=None):
        super(Latent, self).__init__()
        
        self.norm = norm
        if self.norm is None:
            print("No normalizaiton for mid2latent")
        elif self.norm == "BN":
            print("use BN for mid2latent")
            self.bn = nn.BatchNorm1d(num_features=output_size)
        else:
            raise NotImplementedError("unknown normalization for mid2latent")
            
        self.input_size = input_size
        self.output_size = output_size
        self.fc = nn.Linear(self.input_size, self.output_size)
        
        nn.init.xavier_uniform_(self.fc.weight)
        
    def forward(self, x):
        x = self.fc(x)
        if self.norm == "BN":
            x = self.bn(x)
        return x
        
        
        
class VarLatent(nn.Module):
    """
    Adapt from VarLambda of vrae.py
    """
    def __init__(self, input_size, output_size, norm=None):
        super(VarLatent, self).__init__()
        
        print("VarLatent")
        
        self.norm = norm
        if self.norm is None:
            print("No normalizaiton for mid2latent")
        elif self.norm == "BN":
            print("use BN for mid2latent")
            self.bn = nn.BatchNorm1d(num_features=output_size)
        else:
            raise NotImplementedError("unknown normalization for mid2latent")
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.input2mean = nn.Linear(self.input_size, self.output_size)
        self.input2logvar = nn.Linear(self.input_size, self.output_size)
        
        nn.init.xavier_uniform_(self.input2mean.weight)
        nn.init.xavier_uniform_(self.input2logvar.weight)
        
    def forward(self, x):
        self.mean = self.input2mean(x)
        self.logvar = self.input2logvar(x)
        
        if self.training: # nn.Module attribute
            std = torch.exp(0.5*self.logvar)
            eps = torch.randn_like(std)
            self.mean = eps.mul(std).add_(self.mean)
        
        if self.norm == "BN":
            self.mean = self.bn(self.mean)
            
        return self.mean
        
class ConcatPathway(nn.Module):
    """
    Similar to ConcatNestedMemory, but uses a single lstm for each receptor
    Pathway component:
    0. Eheader network
    1. Each receptor in receptor list processes the input signal into corresponding sensitive frequency
    2. The stimuli of different receptors are concatenated, using FA/SA as the pacemaker
    3. The concatenated signal is fed into EncoderNeuron, to obtain the hidden feature h_n
    4. The hidden feature h_n is mapped to a latent representation z, remapped to h_n_hat
    5. The latent representation z is fed into classifier
    6. The h_n_hat is (repeated multiple times and) fed into CatDecoder
    7. The reconstruction output of DecoderNeuron can used for recon loss, paired with input to EncoderNeuron
    """
    
    # currently assume same setting of dropout, bidirectoinal for all submodules
    # blocks are divided into receptor block and mid_block for encoder/decoder neuron
    def __init__(self,
                 number_of_features=19, 
                 num_class=20,
                 x_freq=100,
                 header="MLP",
                 receptor_input_size=None,
                 receptor_freq=[100, 50],
                 receptor_hidden_size=[10, 10],
                 receptor_hidden_layer_depth=[1, 1],
                 dropout=0, 
                 bidirectional=False, 
                 receptor_block="LSTM", 
                 mid_hidden_size=150,
                 mid_hidden_layer_depth=1,
                 mid_block="LSTM",
                 mid_lmu_order=5,
                 latent_length=40,
                 residual=0,
                 device="cpu",
                 var=0,
                 eheader_norm=None,
                 receptor_norm=None,
                 mid_norm=None):
        
        super(ConcatPathway, self).__init__()
        
        self.device = device
        self.header = header
        self.var=int(var)
        
        # Step 0
        # current design: one header for spatial compression, then split to different receptor for temporal compression.
        if self.header == "MLP":
            assert receptor_input_size is not None, "receptor_input_size is required for MLP header"
            self.eheader = MLP(input_size=number_of_features, output_size=receptor_input_size, norm=eheader_norm)
        elif self.header == "CNN":
            assert receptor_input_size == 18, "receptor_input_size is {}, incompatible with current CNN header setting".format(receptor_input_size)
            assert number_of_features == 60, "number_of_features is {}, incompatible with current CNN header setting".format(number_of_features)
            self.eheader = CNN(C=1, H=6, W=10, output_size=receptor_input_size, norm=eheader_norm)
        else:
            raise NotImplementedError("{} header type is not implemented yet".format(self.header))
        
        # Step 1
        self.num_receptor = len(receptor_freq)
        self.receptor_freq = receptor_freq
        self.x_freq = x_freq
        self.receptors = []
        for i in range(self.num_receptor):
            self.receptors.append(Receptor(receptor_freq[i],
                                           receptor_input_size,
                                           receptor_hidden_size[i],
                                           receptor_hidden_layer_depth[i],
                                           dropout=dropout,
                                           bidirectional=bidirectional,
                                           block=receptor_block, 
                                           residual=residual, 
                                           device=device,
                                           norm=receptor_norm))
        self.receptors = nn.ModuleList(self.receptors)
        
        # Step 2,3
        if mid_block == "LSTM":
            self.EncoderNeuron = nn.LSTM(input_size=sum(receptor_hidden_size),
                                        hidden_size=mid_hidden_size,
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            assert mid_lmu_order is not None, "cannot have None for lmu order"
            print("order {} for mid_block lmu".format(mid_lmu_order))
            self.EncoderNeuron = LMU(input_size=sum(receptor_hidden_size),
                                    hidden_size=mid_hidden_size,
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Encoder Neuron yet".format(mid_block))
        
        # Step 4,5
        if self.var:
            self.mid2latent = VarLatent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)
        else:
            self.mid2latent = Latent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)

        self.classifier = nn.Linear(latent_length, num_class)
        self.latent2mid = nn.Linear(latent_length, mid_hidden_size)
        nn.init.xavier_uniform_(self.latent2mid.weight)
        
        # Step 6
        if mid_block == "LSTM":
            self.DecoderNeuron = nn.LSTM(input_size=mid_hidden_size,
                                        hidden_size=sum(receptor_hidden_size),
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            self.DecoderNeuron = LMU(input_size=mid_hidden_size,
                                    hidden_size=sum(receptor_hidden_size),
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Decoder Neuron yet".format(mid_block))       
        
    def forward(self, x):
#         print("enter ConcatPathway, check input size {}".format(x.shape)) # [32, 19, 400] or [32, 6, 10, 75]

        # Step 0
        # after header processing, feature should be at the last dimension
        if self.header == "MLP":
            x = x.permute(0, 2, 1)
            eheader_output = self.eheader(x)
        elif self.header == "CNN":
            x = x.permute(0, 3, 1, 2).unsqueeze(2) # [32, 75, 1, 6, 10]
            batch_size, seq_len, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
            x, mp_indices = self.eheader(x)
            eheader_output = x.reshape(batch_size, seq_len, -1)
#         print("eheader_output", eheader_output.shape)
        
        # Step 1
        # duplicate the eheader_output, fed into different receptors
        # Note: All incoming gradients to the cloned tensor will be propgated to the original one, see https://discuss.pytorch.org/t/how-does-clone-interact-with-backpropagation/8247/6
        receptor_outputs = []
        for i in range(self.num_receptor):
            receptor_input = eheader_output.clone()
            receptor_output = self.receptors[i](receptor_input, self.x_freq)
            receptor_outputs.append(receptor_output)
            
        # Step 2
        encoder_input, max_seq_len = self.process_encoder_neuron_data(receptor_outputs)
#         print("encoder_input", encoder_input.shape)
        
        # Step 3 
        enc_r_out, (enc_h_n, enc_c_n) = self.EncoderNeuron(encoder_input)
        
        # Step 4
        latent = self.mid2latent(enc_h_n)
        
        # Step 5
        classifier_output = self.classifier(latent)
        
        # Step 6
        decoder_input = self.latent2mid(latent)
#         print("before repeat {}".format(decoder_input.shape)) # [1, 32, 150]
        decoder_input_repeat = decoder_input.repeat(max_seq_len, 1, 1).transpose_(0, 1)
        dec_r_out, (dec_h_n, dec_c_n) = self.DecoderNeuron(decoder_input_repeat)
        
        return encoder_input, dec_r_out, enc_h_n, decoder_input, latent, classifier_output.squeeze()
        
    def process_encoder_neuron_data(self, receptor_outputs):
        assert self.num_receptor == len(receptor_outputs), "{} receptors have {} receptor_encoder output {}".format(self.num_receptor, len(receptor_outputs))
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
#             print("receptor {} has input size {}".format(i, receptor_output.shape))
        max_seq_len = max([receptor_outputs[i].shape[1] for i in range(len(receptor_outputs))])
        receptor_outputs_new = []
        # tile 0 
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
            batch_size, seq_len, hidden_size = receptor_output.shape
            multiplier = max_seq_len/seq_len
            receptor_output_new = torch.zeros(batch_size, max_seq_len, hidden_size)
            for j in range(seq_len):
                receptor_output_new[:, int(j*multiplier), :] = receptor_output[:, j, :]
            receptor_outputs_new.append(receptor_output_new)
        receptor_outputs_new = torch.cat(receptor_outputs_new, dim=-1)
#         print("inner_r_outs_new", inner_r_outs_new.shape) # [32, 400, 300]
#         print(inner_r_outs[0][-1, :6, -1])
#         print(inner_r_outs[1][-1, :3, 0])
#         print(inner_r_outs_new[-1,:6,149:151])
        return receptor_outputs_new.to(self.device), max_seq_len    

       
class ConcatPathway_property_layer(nn.Module):
    """
    Similar to ConcatNestedMemory, but uses a single lstm for each receptor
    Pathway component:
    0. Eheader network
    1. Each receptor in receptor list processes the input signal into corresponding sensitive frequency
    2. The stimuli of different receptors are concatenated, using FA/SA as the pacemaker
    3. The concatenated signal is fed into EncoderNeuron, to obtain the hidden feature h_n
    4. The hidden feature h_n is mapped to a latent representation z, remapped to h_n_hat
    5. The latent representation z is fed into classifier
    6. The h_n_hat is (repeated multiple times and) fed into CatDecoder
    7. The reconstruction output of DecoderNeuron can used for recon loss, paired with input to EncoderNeuron
    """
    
    # currently assume same setting of dropout, bidirectoinal for all submodules
    # blocks are divided into receptor block and mid_block for encoder/decoder neuron
    def __init__(self,
                 number_of_features=19, 
                 num_class=20,
                 x_freq=100,
                 header="MLP",
                 receptor_input_size=None,
                 receptor_freq=[100, 50],
                 receptor_hidden_size=[10, 10],
                 receptor_hidden_layer_depth=[1, 1],
                 dropout=0, 
                 bidirectional=False, 
                 receptor_block="LSTM", 
                 mid_hidden_size=150,
                 mid_hidden_layer_depth=1,
                 mid_block="LSTM",
                 mid_lmu_order=5,
                 latent_length=40,
                 residual=0,
                 device="cpu",
                 var=0,
                 eheader_norm=None,
                 receptor_norm=None,
                 mid_norm=None):
        
        super(ConcatPathway_property_layer, self).__init__()
        
        self.device = device
        self.header = header
        self.var=int(var)
        
        # Step 0
        # current design: one header for spatial compression, then split to different receptor for temporal compression.
        if self.header == "MLP":
            assert receptor_input_size is not None, "receptor_input_size is required for MLP header"
            self.eheader = MLP(input_size=number_of_features, output_size=receptor_input_size, norm=eheader_norm)
        elif self.header == "CNN":
            assert receptor_input_size == 18, "receptor_input_size is {}, incompatible with current CNN header setting".format(receptor_input_size)
            assert number_of_features == 60, "number_of_features is {}, incompatible with current CNN header setting".format(number_of_features)
            self.eheader = CNN(C=1, H=6, W=10, output_size=receptor_input_size, norm=eheader_norm)
        else:
            raise NotImplementedError("{} header type is not implemented yet".format(self.header))
        
        # Step 1
        self.num_receptor = len(receptor_freq)
        self.receptor_freq = receptor_freq
        self.x_freq = x_freq
        self.receptors = []
        for i in range(self.num_receptor):
            self.receptors.append(Receptor(receptor_freq[i],
                                           receptor_input_size,
                                           receptor_hidden_size[i],
                                           receptor_hidden_layer_depth[i],
                                           dropout=dropout,
                                           bidirectional=bidirectional,
                                           block=receptor_block, 
                                           residual=residual, 
                                           device=device,
                                           norm=receptor_norm))
        self.receptors = nn.ModuleList(self.receptors)
        
        # Step 2,3
        if mid_block == "LSTM":
            self.EncoderNeuron = nn.LSTM(input_size=sum(receptor_hidden_size),
                                        hidden_size=mid_hidden_size,
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            assert mid_lmu_order is not None, "cannot have None for lmu order"
            print("order {} for mid_block lmu".format(mid_lmu_order))
            self.EncoderNeuron = LMU(input_size=sum(receptor_hidden_size),
                                    hidden_size=mid_hidden_size,
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Encoder Neuron yet".format(mid_block))
        
        # Step 4,5
        if self.var:
            self.mid2latent = VarLatent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)
        else:
            self.mid2latent = Latent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)

        self.classifier = nn.Linear(latent_length, num_class)
        self.latent2mid = nn.Linear(latent_length, mid_hidden_size)
        nn.init.xavier_uniform_(self.latent2mid.weight)
        
        # Step 6
        if mid_block == "LSTM":
            self.DecoderNeuron = nn.LSTM(input_size=mid_hidden_size,
                                        hidden_size=sum(receptor_hidden_size),
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            self.DecoderNeuron = LMU(input_size=mid_hidden_size,
                                    hidden_size=sum(receptor_hidden_size),
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Decoder Neuron yet".format(mid_block))      
        self.stiff_layer = nn.Linear(1, 1)
        self.rough_layer = nn.Linear(1, 1) 
        
    def forward(self, x):
#         print("enter ConcatPathway, check input size {}".format(x.shape)) # [32, 19, 400] or [32, 6, 10, 75]

        # Step 0
        # after header processing, feature should be at the last dimension
        if self.header == "MLP":
            x = x.permute(0, 2, 1)
            eheader_output = self.eheader(x)
        elif self.header == "CNN":
            x = x.permute(0, 3, 1, 2).unsqueeze(2) # [32, 75, 1, 6, 10]
            batch_size, seq_len, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
            x, mp_indices = self.eheader(x)
            eheader_output = x.reshape(batch_size, seq_len, -1)
#         print("eheader_output", eheader_output.shape)
        
        # Step 1
        # duplicate the eheader_output, fed into different receptors
        # Note: All incoming gradients to the cloned tensor will be propgated to the original one, see https://discuss.pytorch.org/t/how-does-clone-interact-with-backpropagation/8247/6
        receptor_outputs = []
        for i in range(self.num_receptor):
            receptor_input = eheader_output.clone()
            receptor_output = self.receptors[i](receptor_input, self.x_freq)
            receptor_outputs.append(receptor_output)
            
        # Step 2
        encoder_input, max_seq_len = self.process_encoder_neuron_data(receptor_outputs)
#         print("encoder_input", encoder_input.shape)
        
        # Step 3 
        enc_r_out, (enc_h_n, enc_c_n) = self.EncoderNeuron(encoder_input)
        
        # Step 4
        latent = self.mid2latent(enc_h_n)
        
        # Step 5
        classifier_output = self.classifier(latent)
        # print(latent.size())

        stiff_latent = latent[:, :, 0].reshape(32, 1)
        rough_latent = latent[:, :, 1].reshape(32, 1)
        s = self.stiff_layer(stiff_latent)
        s = torch.squeeze(s)
        r = self.rough_layer(rough_latent)
        r = torch.squeeze(r)
        
        # Step 6
        decoder_input = self.latent2mid(latent)
#         print("before repeat {}".format(decoder_input.shape)) # [1, 32, 150]
        decoder_input_repeat = decoder_input.repeat(max_seq_len, 1, 1).transpose_(0, 1)
        dec_r_out, (dec_h_n, dec_c_n) = self.DecoderNeuron(decoder_input_repeat)
        
        return encoder_input, dec_r_out, enc_h_n, decoder_input, latent, classifier_output.squeeze(), s, r
        
    def process_encoder_neuron_data(self, receptor_outputs):
        assert self.num_receptor == len(receptor_outputs), "{} receptors have {} receptor_encoder output {}".format(self.num_receptor, len(receptor_outputs))
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
#             print("receptor {} has input size {}".format(i, receptor_output.shape))
        max_seq_len = max([receptor_outputs[i].shape[1] for i in range(len(receptor_outputs))])
        receptor_outputs_new = []
        # tile 0 
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
            batch_size, seq_len, hidden_size = receptor_output.shape
            multiplier = max_seq_len/seq_len
            receptor_output_new = torch.zeros(batch_size, max_seq_len, hidden_size)
            for j in range(seq_len):
                receptor_output_new[:, int(j*multiplier), :] = receptor_output[:, j, :]
            receptor_outputs_new.append(receptor_output_new)
        receptor_outputs_new = torch.cat(receptor_outputs_new, dim=-1)
#         print("inner_r_outs_new", inner_r_outs_new.shape) # [32, 400, 300]
#         print(inner_r_outs[0][-1, :6, -1])
#         print(inner_r_outs[1][-1, :3, 0])
#         print(inner_r_outs_new[-1,:6,149:151])
        return receptor_outputs_new.to(self.device), max_seq_len    





class ConcatPathwayEnsemble(nn.Module):
    """
    Similar to ConcatPathway, but allows multiple sensor data to be trained together.
    Modify a single header to a header dict, including the information of receptor freq and ihs inside the dict
    """
    
    # currently assume same setting of dropout, bidirectoinal for all submodules
    # blocks are divided into receptor block and mid_block for encoder/decoder neuron
    def __init__(self,
                 num_class=20,
                 header_dict = {"MLP": {"number_of_features": 19, "x_freq": 100, "receptor_freq": [20, 10], "receptor_hidden_size": [100, 200], "receptor_input_size":18}, "CNN": {"number_of_features": 60, "x_freq": 50, "receptor_freq": [20, 10], "receptor_hidden_size": [100, 200], "receptor_input_size": 18}},
                 receptor_hidden_layer_depth=1,
                 dropout=0, 
                 bidirectional=False, 
                 receptor_block="LSTM", 
                 mid_hidden_size=150,
                 mid_hidden_layer_depth=1,
                 mid_block="LSTM",
                 mid_lmu_order=5,
                 latent_length=40,
                 residual=0,
                 device="cpu",
                 var=0,
                 eheader_norm=None,
                 receptor_norm=None,
                 mid_norm=None):
        
        super(ConcatPathwayEnsemble, self).__init__()
        
        self.device = device
        self.header_dict = header_dict
        self.var=int(var)
        
        # Step 0
        # parse the header_dict and create corresponding receptors
        print("required {} headers: {}".format(len(header_dict.keys()), header_dict.keys()))
        self.num_eheader = len(header_dict.keys())
        self.headers = nn.ModuleDict({})
        self.num_receptor = {}
        self.receptor_freq = {}
        self.receptors = {}
        self.x_freq = {}
        receptor_hidden_size_sum = None
        for header, receptors in header_dict.items():
            print("header")
            print(header)
            print("receptors")
            print(receptors)
            if header == "MLP":
                self.headers["B"] = MLP(input_size=receptors["number_of_features"], output_size=receptors["receptor_input_size"], norm=eheader_norm)
                tag = "B"

            elif header == "CNN":
                assert receptors["receptor_input_size"] == 18, "receptor_input_size is {}, incompatible with current CNN header setting".format(receptors["receptor_input_size"])
                assert receptors["number_of_features"] == 60, "number_of_features is {}, incompatible with current CNN header setting".format(receptors["number_of_features"])
                self.headers["I"] = CNN(C=1, H=6, W=10, output_size=receptors["receptor_input_size"], norm=eheader_norm)
                tag = "I"
            else:
                raise NotImplementedError("{} header type is not implemented yet".format(self.header))
        
            # Step 1
            self.x_freq[tag] = receptors["x_freq"]
            self.num_receptor[tag] = len(receptors["receptor_freq"])
            self.receptor_freq[tag] = receptors["receptor_freq"]
            self.receptors[tag] = []

            for i in range(self.num_receptor[tag]):
                self.receptors[tag].append(Receptor(receptors["receptor_freq"][i],
                                               receptors["receptor_input_size"],
                                               receptors["receptor_hidden_size"][i],
                                               receptor_hidden_layer_depth,
                                               dropout=dropout,
                                               bidirectional=bidirectional,
                                               block=receptor_block, 
                                               residual=residual, 
                                               device=device,
                                               norm=receptor_norm))
            self.receptors[tag] = nn.ModuleList(self.receptors[tag]).to(self.device)
            if receptor_hidden_size_sum is None:
                receptor_hidden_size_sum = sum(receptors["receptor_hidden_size"])
            else:
                assert receptor_hidden_size_sum == sum(receptors["receptor_hidden_size"]), "Inconsistent receptor hidden size sum for mid_block input"
        
        self.headers = self.headers.to(self.device)
        
        # Step 2,3
        if mid_block == "LSTM":
            self.EncoderNeuron = nn.LSTM(input_size=receptor_hidden_size_sum,
                                        hidden_size=mid_hidden_size,
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            assert mid_lmu_order is not None, "cannot have None for lmu order"
            print("order {} for mid_block lmu".format(mid_lmu_order))
            self.EncoderNeuron = LMU(input_size=sum(receptor_hidden_size),
                                    hidden_size=mid_hidden_size,
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Encoder Neuron yet".format(mid_block))
        
        # Step 4,5
        if self.var:
            self.mid2latent = VarLatent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)
        else:
            self.mid2latent = Latent(input_size=mid_hidden_size, output_size=latent_length, norm=mid_norm)

        self.classifier = nn.Linear(latent_length, num_class)
        self.latent2mid = nn.Linear(latent_length, mid_hidden_size)
        nn.init.xavier_uniform_(self.latent2mid.weight)
        
        # Step 6
        if mid_block == "LSTM":
            self.DecoderNeuron = nn.LSTM(input_size=mid_hidden_size,
                                        hidden_size=receptor_hidden_size_sum,
                                        num_layers=mid_hidden_layer_depth,
                                        batch_first=True, dropout=dropout)
        elif mid_block == "LMU":
            self.DecoderNeuron = LMU(input_size=mid_hidden_size,
                                    hidden_size=receptor_hidden_size_sum,
                                    order=mid_lmu_order,
                                    batch_first=True,
                                    device=self.device)
        else:
            raise NotImplementedError("block {} not implemented for Decoder Neuron yet".format(mid_block))       
        
    def forward(self, x, tag):
        
        # add input information tag
        
        # Step 0
        # after header processing, feature should be at the last dimension
        if tag == "B":
            x = x.permute(0, 2, 1)
            eheader_output = self.headers[tag](x)
        elif tag == "I":
            x = x.permute(0, 3, 1, 2).unsqueeze(2) # [32, 75, 1, 6, 10]
            batch_size, seq_len, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
            x, mp_indices = self.headers[tag](x)
            eheader_output = x.reshape(batch_size, seq_len, -1)
        else:
            raise NotImplementedError("Unknow tag {}".format(tag))
        
        
        # Step 1
        # duplicate the eheader_output, fed into different receptors
        # Note: All incoming gradients to the cloned tensor will be propgated to the original one, see https://discuss.pytorch.org/t/how-does-clone-interact-with-backpropagation/8247/6
        receptor_outputs = []
        for i in range(self.num_receptor[tag]):
            receptor_input = eheader_output.clone()
            receptor_output = self.receptors[tag][i](receptor_input, self.x_freq[tag])
            receptor_outputs.append(receptor_output)
            
        # Step 2
        encoder_input, max_seq_len = self.process_encoder_neuron_data(receptor_outputs)
#         print("encoder_input", encoder_input.shape)
        
        # Step 3 
        enc_r_out, (enc_h_n, enc_c_n) = self.EncoderNeuron(encoder_input)
        
        # Step 4
        latent = self.mid2latent(enc_h_n)
        
        # Step 5
        classifier_output = self.classifier(latent)
        
        # Step 6
        decoder_input = self.latent2mid(latent)
#         print("before repeat {}".format(decoder_input.shape)) # [1, 32, 150]
        decoder_input_repeat = decoder_input.repeat(max_seq_len, 1, 1).transpose_(0, 1)
        dec_r_out, (dec_h_n, dec_c_n) = self.DecoderNeuron(decoder_input_repeat)
        
        return encoder_input, dec_r_out, enc_h_n, decoder_input, latent, classifier_output.squeeze()
        
    def process_encoder_neuron_data(self, receptor_outputs):
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
#             print("receptor {} has input size {}".format(i, receptor_output.shape))
        max_seq_len = max([receptor_outputs[i].shape[1] for i in range(len(receptor_outputs))])
        receptor_outputs_new = []
        # tile 0 
        for i in range(len(receptor_outputs)):
            receptor_output = receptor_outputs[i]
            batch_size, seq_len, hidden_size = receptor_output.shape
            multiplier = max_seq_len/seq_len
            receptor_output_new = torch.zeros(batch_size, max_seq_len, hidden_size)
            for j in range(seq_len):
                receptor_output_new[:, int(j*multiplier), :] = receptor_output[:, j, :]
            receptor_outputs_new.append(receptor_output_new)
        receptor_outputs_new = torch.cat(receptor_outputs_new, dim=-1)
#         print("inner_r_outs_new", inner_r_outs_new.shape) # [32, 400, 300]
#         print(inner_r_outs[0][-1, :6, -1])
#         print(inner_r_outs[1][-1, :3, 0])
#         print(inner_r_outs_new[-1,:6,149:151])
        return receptor_outputs_new.to(self.device), max_seq_len    
    

    