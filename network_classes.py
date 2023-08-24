# Simple fully connected network

import torch

class fc_nn (torch.nn.Module):
    
    def __init__(self, no_input_nodes, use_input_batchnorm, no_layers, l_nodes,
                 l_activation_funcs, l_use_batch_norm, output_nodes, output_activation):
        
        super().__init__()
        
        # Should an input normalization layer be learned?
        self.use_input_batchnorm = use_input_batchnorm
        
        if use_input_batchnorm:
            self.input_batchnorm = torch.nn.BatchNorm1d(no_input_nodes)
        
        # Define the number of layers, nodes per layer, the activation function
        # and decide whether to use batch normalization after each layer or not
        self.no_layers = no_layers
        self.nodes = [no_input_nodes] + l_nodes if type(l_nodes) == list else [no_input_nodes] + [l_nodes for m in range(no_layers)]
        self.activation_func = l_activation_funcs if type(l_activation_funcs) == list else [l_activation_funcs for m in range(no_layers)]
        self.use_batch_norm = l_use_batch_norm if type(l_use_batch_norm) == list else [l_use_batch_norm for m in range(no_layers)]
        
        # Instantiate the layers using the specified variables as well as the output layer
        self.fcn_layers = torch.nn.ModuleList([fcn_layer(self.nodes[i], self.nodes[i+1], self.use_batch_norm[i], self.activation_func[i]) for i in range(no_layers)])
        self.output_layer = torch.nn.Linear(self.nodes[-1], output_nodes)
        self.output_activation = output_activation
        
    
    # Call of the Model applying to input x
    def forward(self, x):
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(self.no_layers):
            x = self.fcn_layers[m](x)
        
        return self.output_activation(self.output_layer(x)).squeeze()
    
    def get_features(self, x, layer_idx = None):
        if layer_idx == None:
            layer_idx = self.no_layers
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(layer_idx):
            x = self.fcn_layers[m](x)
        
        return x

# Class for a single layer, the basic block of a vanilla NN
class fcn_layer (torch.nn.Module):
    
    def __init__(self, input_nodes, output_nodes, use_batch_norm, activation_func):
        
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        self.layer = torch.nn.Linear(input_nodes, output_nodes)
        
        if use_batch_norm:
            self.norm = torch.nn.BatchNorm1d(output_nodes)
        
        self.act_func = activation_func
        
    def forward(self, x):
        
        x = self.layer(x)
        if self.use_batch_norm:
            x = self.norm(x)
        
        x = self.act_func(x)
        
        return x

# FCN Layer adapted to the new learnable 
# activation function (= small network w/ 1 input node and 1 output node)
class actvf_fcn_layer (torch.nn.Module):
    
    def __init__(self, input_nodes, output_nodes, use_batch_norm, activation_func):
        
        super().__init__()
        
        self.use_batch_norm = use_batch_norm
        
        self.layer = torch.nn.Linear(input_nodes, output_nodes)
        
        if use_batch_norm:
            self.norm = torch.nn.BatchNorm1d(output_nodes)
        
        self.act_func = activation_func
        
    def forward(self, x):
        
        x = self.layer(x)
        if self.use_batch_norm:
            x = self.norm(x)
        
        orig_shape = x.shape
        x = self.act_func(x.reshape(-1,1)) # Create new dimension to perform same
                                           # activation transformation on all nodes
        
        return x.squeeze().reshape(orig_shape)               # Reduce to the original dimensions

    
    
# Simple fully connected network w/ learnable activation function
class actvf_fc_nn (torch.nn.Module):
    
    def __init__(self, no_input_nodes, use_input_batchnorm, no_layers, l_nodes,
                 base_activation, actvf_layers, actvf_nodes, l_use_batch_norm, output_nodes, output_activation):
        
        super().__init__()
        
        # Should an input normalization layer be learned?
        self.use_input_batchnorm = use_input_batchnorm
        
        if use_input_batchnorm:
            self.input_batchnorm = torch.nn.BatchNorm1d(no_input_nodes)
        
        # Define the number of layers, nodes per layer, the activation function
        # and decide whether to use batch normalization after each layer or not
        self.no_layers = no_layers
        self.nodes = [no_input_nodes] + l_nodes if type(l_nodes) == list else [no_input_nodes] + [l_nodes for m in range(no_layers)]
        self.use_batch_norm = l_use_batch_norm if type(l_use_batch_norm) == list else [l_use_batch_norm for m in range(no_layers)]
        
        # Define activation function as small nn to get a learnable activation function
        self.activation_func = fc_nn(1, False, actvf_layers, actvf_nodes, base_activation, True, 1, torch.nn.Identity())
        
        # Instantiate the layers using the specified variables as well as the output layer
        self.fcn_layers = torch.nn.ModuleList([actvf_fcn_layer(self.nodes[i], self.nodes[i+1], self.use_batch_norm[i], self.activation_func) for i in range(no_layers)])
        self.output_layer = torch.nn.Linear(self.nodes[-1], output_nodes)
        self.output_activation = output_activation
    
    # Call of the Model applying to input x
    def forward(self, x):
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(self.no_layers):
            x = self.fcn_layers[m](x)
        
        return self.output_activation(self.output_layer(x)).squeeze()
    
    def get_features(self, x, layer_idx = None):
        if layer_idx == None:
            layer_idx = self.no_layers
        
        if self.use_input_batchnorm:
            x = self.input_batchnorm(x)
        
        for m in range(layer_idx):
            x = self.fcn_layers[m](x)
        
        return x
    
    def train(self, do_train = True, actv_eval = False):
        
        super().train(do_train)
        if actv_eval:
            self.activation_func.eval()

        
        