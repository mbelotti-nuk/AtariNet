import torch
import torch.nn as nn
import numpy as np


# This class defines the dueling DQN network structure
class Dueling_DQNnet(nn.Module):
    def __init__(self, input_dim, out_dim, filename, n_frames=4, init_weights=True):
        super(Dueling_DQNnet, self).__init__()
        self.input_dim = [ input_dim[0], input_dim[1], input_dim[2] ] 
        channels = n_frames
        self.input_dim[0] = channels

        # 3 conv layers, all with relu activations, first one with maxpool
        self.convoluation_layer = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,bias=False),
            nn.ReLU(True)
        )

        # Calculate output dimensions for linear layer
        conv_out_dim = self.conv_output_dim()
        lin1_out_dim = 512

        # Two fully connected layers with one relu activation
        # for estimating value of the state and the
        # advantage of each action
        self.value= nn.Sequential(
            nn.Linear(conv_out_dim, lin1_out_dim),
            nn.ReLU(),
            nn.Linear(lin1_out_dim, 1)
        )

        self.adv = nn.Sequential(
            nn.Linear(conv_out_dim, lin1_out_dim),
            nn.ReLU(),
            nn.Linear(lin1_out_dim, out_dim)
        )
        # Save filename for saving model
        self.filename = filename
        if init_weights:
            self._initialize_weights()

    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.convoluation_layer(x)
        return int(np.prod(x.shape))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)

    # Performs forward pass through the network, returns action values
    def forward(self, x) -> torch.tensor:

        x = self.convoluation_layer(x)
        x = x.view(x.shape[0], -1)
    
        adv = self.adv(x) 
        value = self.value(x) 
              
        # combine this 2 values together
        Q_value = value + (adv - torch.mean(adv,dim=1,keepdim=True))
        

        return Q_value

    # Save a model

    def save_model(self):
        with torch.no_grad():
            torch.save(self.state_dict(), 'NET/models/' + self.filename + '.pt')

    # Loads a model
    def load_model(self):
        self.load_state_dict(torch.load('NET/models/' + self.filename + '.pt'))


class DQNnet(nn.Module):
    def __init__(self, input_dim, out_dim, filename, n_frames=4):
        super(DQNnet, self).__init__()
        self.input_dim = [ input_dim[0], input_dim[1], input_dim[2] ] 
        channels = n_frames
        self.input_dim[0] = channels

        # 3 conv layers, all with relu activations, first one with maxpool
        self.convoluation_layer = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=2, bias=False),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,bias=False),
            nn.ReLU(True)
        )

        # Calculate output dimensions for linear layer
        conv_out_dim = self.conv_output_dim()
        lin1_out_dim = 512


        self.linear = nn.Sequential(
            nn.Linear(conv_out_dim, lin1_out_dim),
            nn.ReLU(),
            nn.Linear(lin1_out_dim, out_dim)
        )
        # Save filename for saving model
        self.filename = filename


    # Calulates output dimension of conv layers
    def conv_output_dim(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.convoluation_layer(x)
        return int(np.prod(x.shape))
    
    # Performs forward pass through the network, returns action values
    def forward(self, x) -> torch.tensor:

        x = self.convoluation_layer(x)
        x = x.view(x.shape[0], -1)
        Q_value = self.linear(x) 
        return Q_value

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1.0)
            #     nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.constant_(m.bias, 0.0)


    # Save a model
    def save_model(self):
        with torch.no_grad():
            torch.save(self.state_dict(), 'NET/models/' + "DDQN_" +self.filename + '.pt')

    # Loads a model
    def load_model(self, path):
        self.load_state_dict(torch.load(path)) #+ self.filename + '.pt')))