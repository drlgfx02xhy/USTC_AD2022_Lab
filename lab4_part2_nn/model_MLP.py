import torch
import torch.nn as nn
import torch.functional as F

activate_dict = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid
}

class MLP(nn.Module):
    def __init__(self, num_layers: int, activate_function: str, neural_num_list: list, need_bias = True):
        super(MLP, self).__init__()
        assert activate_function in ['relu', 'tanh', 'sigmoid']

        self.num_layers = num_layers
        self.layer_list = neural_num_list
        self.MLP = nn.Sequential()
        self.need_bias = need_bias
        
        layer = 0
        if len(neural_num_list) == 2:
            new_layer = nn.Linear(self.layer_list[layer], self.layer_list[layer+1], bias = self.need_bias)
            self.MLP.add_module("L{} linear layer".format(str(layer)), new_layer)
        else:
            for layer, _ in enumerate(self.layer_list[:-2]):
                new_layer = nn.Linear(self.layer_list[layer], self.layer_list[layer+1], bias = self.need_bias)
                self.MLP.add_module("L{} linear layer".format(str(layer)), new_layer)
                self.MLP.add_module("L{} activate layer".format(str(layer)), activate_dict[activate_function]())
            new_layer = nn.Linear(self.layer_list[layer+1], self.layer_list[layer+2], bias = self.need_bias)
            self.MLP.add_module("L{} linear layer".format(str(layer+1)), new_layer)
    
    def forward(self, x):
        x = x.float()
        return self.MLP(x)