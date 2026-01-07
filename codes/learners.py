import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.activation import ReLU, Sigmoid
from transformers import BertTokenizer, BertModel, AutoModel, AutoTokenizer, RobertaTokenizer, RobertaModel
from deepchem.metalearning.torch_maml import MetaLearner, MAML
import random
import numpy as np
import math

class CensoredRegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, censorship: torch.Tensor) -> torch.Tensor:
        '''  0: exact value
             1: right-censored (> value)
            -1: left-censored (< value) '''
        # Compute squared error for exact values
        squared_errors = (predictions - targets) ** 2

        # Compute loss based on censorship
        loss = torch.where(
            censorship == 0,  # Exact values
            squared_errors,
            torch.where(
                censorship == 1,  # Right-censored: Penalize if predictions exceed targets
                torch.clamp_min(predictions - targets, 0) ** 2,
                torch.clamp_min(targets - predictions, 0) ** 2,  # Left-censored: Penalize if predictions are below targets
            )
        )

        return loss.sum()
    
class CensoredHybridLoss(nn.Module):
    """
    Combines a censored MSE-like loss for regression targets
    with an auxiliary binary classification loss based on 
    transformed target magnitudes.
    """

    def __init__(self):
        super().__init__()

    def censored_mse(self, predictions: torch.Tensor, targets: torch.Tensor, censorship: torch.Tensor) -> torch.Tensor:

        squared_errors = (predictions - targets) ** 2

        censored_loss = torch.where(
            censorship == 0,
            squared_errors,
            torch.where(
                censorship == 1,
                torch.clamp_min(predictions - targets, 0) ** 2,
                torch.clamp_min(targets - predictions, 0) ** 2
            )
        )
        return censored_loss

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, censorship: torch.Tensor) -> torch.Tensor:

        mse_loss = self.censored_mse(predictions, targets, censorship).mean()

        y_prim = torch.tanh(targets)
        x_prim = torch.tanh(predictions)

        binary_labels = (y_prim > 0.5).float()
        bce_loss = F.binary_cross_entropy_with_logits(x_prim, binary_labels)

        loss = mse_loss + bce_loss
        return loss


class MetaMFLearner(MetaLearner):
    def __init__(self, layer_sizes=[1, 40, 20, 1], activation=F.relu, dataset=None, batch_size=10, tasks_dict=None, test_tasks=None):
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.dataset = dataset
        self.task_index = None
        self.tasks_dict = tasks_dict
        self.test_tasks = test_tasks
        self.layers = self._create_layers()
        self.train_indices = []
        self.loss_fn = CensoredRegressionLoss()

    def _create_layers(self):
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            layers.append(layer)
        return torch.nn.ModuleList(layers)

    def compute_model(self, inputs, variables, training):
        device = next(self.parameters()).device
        x, y, censorship = inputs
        x = x.to(device)
        y = y.to(device)
        censorship = censorship.to(device)

        current_variables = variables 
        param_index = 0
        for i, layer in enumerate(self.layers):
            weight = current_variables[param_index]
            bias = current_variables[param_index + 1]
            if i < len(self.layers) - 1:
                x = self.activation(F.linear(x, weight=weight, bias=bias))
            else:
                x = F.linear(x, weight=weight, bias=bias)
            param_index += 2 
        loss = self.loss_fn(x, y, censorship)
        return loss, [x]


    @property
    def variables(self):
        return [param for layer in self.layers for param in layer.parameters()]

    def select_task(self):
        unique_ids = self.tasks_dict.values()
        unique_ids = [x for x in unique_ids if x not in self.test_tasks]
        self.task_index = random.choice(unique_ids)

    def select_task_by_name(self, task_name):
        self.task_index = self.tasks_dict[task_name]

    def get_batch(self):
        task_indices = np.where(self.dataset.ids == self.task_index)[0]
        if len(task_indices) < self.batch_size:
            batch_indices = np.random.choice(task_indices, len(task_indices), replace=False)
        else:
            batch_indices = np.random.choice(task_indices, self.batch_size, replace=False)
        self.train_indices = batch_indices
        x = torch.tensor(self.dataset.X[batch_indices], dtype=torch.float32)
        y = torch.tensor(self.dataset.y[batch_indices], dtype=torch.float32).view(-1, 1)
        censorship = torch.tensor(self.dataset.w[batch_indices], dtype=torch.float32).view(-1, 1)
        return [x, y, censorship]

    def parameters(self):
        for param in self.variables:
            yield param

    def set_params_to_layers_from_list(self, params_list):
        current_params_generator = (param for param in self.parameters())
        for new_param in params_list:
            try:
                old_param = next(current_params_generator)
                if old_param.shape != new_param.shape:
                     raise ValueError(f"Shape mismatch: Expected {old_param.shape}, but got {new_param.shape}")
                old_param.data = new_param.data.to(old_param.device)
            except StopIteration:
                raise ValueError("More parameters in params_list than in the model layers.")
        try:
            next(current_params_generator)
            raise ValueError("Fewer parameters in params_list than in the model layers.")
        except StopIteration:
            pass

class IFMLayer(nn.Module):
    def __init__(self, input_dim, num_frequencies=8, init_sigma=6.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.freqs = nn.Parameter(
            init_sigma * torch.randn(input_dim, num_frequencies)
        )

class MetaIFMMFLearner_classic(MetaLearner):
    def __init__(self, layer_sizes=[1, 40, 20, 1], activation=F.relu, dataset=None, batch_size=10, tasks_dict=None, test_tasks=None, lossfn=None):
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.ifm_freq = 8
        self.ifm_input_dim = self.layer_sizes[0]
        self.ifm = IFMLayer(input_dim=layer_sizes[0], num_frequencies=self.ifm_freq)
        self.ifm_param_count = sum(p.numel() for p in self.ifm.parameters())
        self.layer_sizes[0] = 2 * self.ifm_freq * self.layer_sizes[0]
        self.activation = activation
        self.dataset = dataset
        self.task_index = None
        self.tasks_dict = tasks_dict
        self.test_tasks = test_tasks
        self.layers = self._create_layers()
        self.train_indices = []
        if lossfn:
            self.loss_fn = lossfn()
        else:
            self.loss_fn = CensoredRegressionLoss()

    def _create_layers(self):
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            layers.append(layer)
        return torch.nn.ModuleList(layers)

    def switch_loss(self):
        self.loss_fn = CensoredRegressionLoss()

    def _ifm_forward(self, x, flat_params):
        input_dim = self.ifm.input_dim
        num_freqs = self.ifm.num_frequencies

        freqs = flat_params.view(input_dim, num_freqs)

        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        sin_feats = torch.sin(x_expanded)
        cos_feats = torch.cos(x_expanded)
        out = torch.cat([sin_feats, cos_feats], dim=-1)
        return out.view(batch_size, -1)

    def compute_model(self, inputs, variables, training):
        device = next(self.parameters()).device
        x, y, censorship = inputs
        x = x.to(device)
        y = y.to(device)
        censorship = censorship.to(device)

        ifm_param = variables[0]
        x = self._ifm_forward(x, ifm_param)

        current_variables = variables[1:]
        param_index = 0 
        for i, layer in enumerate(self.layers):
            weight = current_variables[param_index]
            bias = current_variables[param_index + 1]
            if i < len(self.layers) - 1:
                x = self.activation(F.linear(x, weight=weight, bias=bias))
            else:
                x = F.linear(x, weight=weight, bias=bias)
            param_index += 2 

        
        loss = self.loss_fn(x, y, censorship)

        return loss, [x]


    @property
    def variables(self):
        return list(self.ifm.parameters()) + [param for layer in self.layers for param in layer.parameters()]

    def select_task(self):
        unique_ids = self.tasks_dict.values()
        unique_ids = [x for x in unique_ids if x not in self.test_tasks]
        self.task_index = random.choice(unique_ids)

    def select_task_by_name(self, task_name):
        self.task_index = self.tasks_dict[task_name]

    def get_batch(self):
        task_indices = np.where(self.dataset.ids == self.task_index)[0]
        if len(task_indices) < self.batch_size:
            batch_indices = np.random.choice(task_indices, len(task_indices), replace=False)
        else:
            batch_indices = np.random.choice(task_indices, self.batch_size, replace=False)
        self.train_indices = batch_indices
        x = torch.tensor(self.dataset.X[batch_indices], dtype=torch.float32)
        y = torch.tensor(self.dataset.y[batch_indices], dtype=torch.float32).view(-1, 1)
        censorship = torch.tensor(self.dataset.w[batch_indices], dtype=torch.float32).view(-1, 1)

        return [x, y, censorship]

    def parameters(self):
        for param in self.variables:
            yield param

    def set_params_to_layers_from_list(self, params_list):
        current_params_generator = (param for param in self.parameters())
        for new_param in params_list:
            try:
                old_param = next(current_params_generator)
                if old_param.shape != new_param.shape:
                     raise ValueError(f"Shape mismatch: Expected {old_param.shape}, but got {new_param.shape}")
                old_param.data = new_param.data.to(old_param.device)
            except StopIteration:
                raise ValueError("More parameters in params_list than in the model layers.")
        try:
            next(current_params_generator)
            raise ValueError("Fewer parameters in params_list than in the model layers.")
        except StopIteration:
            pass


class MetaIFMMFLearner(MetaLearner):
    def __init__(self, layer_sizes=[1, 40, 20, 1], activation=F.relu, dataset=None, batch_size=10, tasks_dict=None, test_tasks=None):
        self.batch_size = batch_size
        self.layer_sizes = layer_sizes
        self.ifm_freq = 8
        self.ifm_input_dim = self.layer_sizes[0]
        self.ifm = IFMLayer(input_dim=layer_sizes[0], num_frequencies=self.ifm_freq)
        self.ifm_param_count = sum(p.numel() for p in self.ifm.parameters())
        self.layer_sizes[0] = 2 * self.ifm_freq * self.layer_sizes[0]
        self.activation = activation
        self.dataset = dataset
        self.task_index = None 
        self.tasks_dict = tasks_dict
        self.test_tasks = test_tasks
        
        self.latent_dim = self.layer_sizes[1]
        
        self.attention = self._create_attention_module()
        
        self.layers = self._create_layers()
        self.train_indices = []
        self.loss_fn = CensoredRegressionLoss()

    def _create_attention_module(self):

        attention_layers = nn.ModuleList([
            nn.Linear(self.latent_dim + 1, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Sigmoid()
        ])
        return attention_layers

    def _create_layers(self):
        layers = []
        for i in range(len(self.layer_sizes) - 1):
            layer = torch.nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1])
            layers.append(layer)
        return torch.nn.ModuleList(layers)

    def _ifm_forward(self, x, flat_params):
        input_dim = self.ifm.input_dim
        num_freqs = self.ifm.num_frequencies

        freqs = flat_params.view(input_dim, num_freqs)

        batch_size = x.size(0)
        x_expanded = x.unsqueeze(-1) * freqs.unsqueeze(0) * 2 * math.pi
        sin_feats = torch.sin(x_expanded)
        cos_feats = torch.cos(x_expanded)
        out = torch.cat([sin_feats, cos_feats], dim=-1)
        return out.view(batch_size, -1)

    def _compute_attention(self, z, task_id, attention_params):

        device = z.device
        batch_size = z.size(0)

        task_embedding = torch.full((batch_size, 1), float(task_id) / 100.0, device=device)
        
        attention_input = torch.cat([z, task_embedding], dim=-1)
        
        param_idx = 0
        x = attention_input
        for i, layer in enumerate(self.attention):
            if isinstance(layer, nn.Linear):
                weight = attention_params[param_idx]
                bias = attention_params[param_idx + 1]
                x = F.linear(x, weight=weight, bias=bias)
                param_idx += 2
            elif isinstance(layer, nn.ReLU):
                x = F.relu(x)
            elif isinstance(layer, nn.Sigmoid):
                x = torch.sigmoid(x)
        
        return x

    def compute_model(self, inputs, variables, training):
        device = next(self.parameters()).device
        x, y, censorship = inputs
        x = x.to(device)
        y = y.to(device)
        censorship = censorship.to(device)

        ifm_param = variables[0]
        x = self._ifm_forward(x, ifm_param)

        attention_param_count = sum(2 for layer in self.attention if isinstance(layer, nn.Linear))
        attention_params = variables[1:1 + attention_param_count]
        
        current_variables = variables[1 + attention_param_count:]
        
        param_index = 0
        for i, layer in enumerate(self.layers):
            weight = current_variables[param_index]
            bias = current_variables[param_index + 1]
            
            if i == 0:
                x = self.activation(F.linear(x, weight=weight, bias=bias))
                z = x                 
                if self.task_index is not None:
                    attention_weights = self._compute_attention(z, self.task_index, attention_params)
                    x = attention_weights * z
            elif i < len(self.layers) - 1:
                x = self.activation(F.linear(x, weight=weight, bias=bias))
            else:
                x = F.linear(x, weight=weight, bias=bias)
            
            param_index += 2
        
        loss = self.loss_fn(x, y, censorship)

        return loss, [x]

    @property
    def variables(self):
        return (list(self.ifm.parameters()) + 
                list(self.attention.parameters()) + 
                [param for layer in self.layers for param in layer.parameters()])

    def select_task(self):
        unique_ids = self.tasks_dict.values()
        unique_ids = [x for x in unique_ids if x not in self.test_tasks]
        self.task_index = random.choice(unique_ids)

    def select_task_by_name(self, task_name):
        self.task_index = self.tasks_dict[task_name]

    def get_batch(self):
        task_indices = np.where(self.dataset.ids == self.task_index)[0]
        if len(task_indices) < self.batch_size:
            batch_indices = np.random.choice(task_indices, len(task_indices), replace=False)
        else:
            batch_indices = np.random.choice(task_indices, self.batch_size, replace=False)
        self.train_indices = batch_indices
        x = torch.tensor(self.dataset.X[batch_indices], dtype=torch.float32)
        y = torch.tensor(self.dataset.y[batch_indices], dtype=torch.float32).view(-1, 1)
        censorship = torch.tensor(self.dataset.w[batch_indices], dtype=torch.float32).view(-1, 1)
        
        return [x, y, censorship]

    def parameters(self):
        for param in self.variables:
            yield param

    def set_params_to_layers_from_list(self, params_list):
        current_params_generator = (param for param in self.parameters())
        for new_param in params_list:
            try:
                old_param = next(current_params_generator)
                if old_param.shape != new_param.shape:
                     raise ValueError(f"Shape mismatch: Expected {old_param.shape}, but got {new_param.shape}")
                old_param.data = new_param.data.to(old_param.device)
            except StopIteration:
                raise ValueError("More parameters in params_list than in the model layers.")
        try:
            next(current_params_generator)
            raise ValueError("Fewer parameters in params_list than in the model layers.")
        except StopIteration:
            pass