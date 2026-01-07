"""Attentive Model-Agnostic Meta-Learning (Att-MAML) modification of Model-Agnostic Meta-Learning (MAML) algorithm for low data learning."""

import os
import shutil
import tempfile
import time
import random 
import numpy as np
from deepchem.models.optimizers import Optimizer, Adam, GradientDescent, LearningRateSchedule
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union
from deepchem.utils.typing import OneOrMany

try:
    from deepchem.metalearning import MetaLearner
    from deepchem.metalearning.torch_maml import MAML
    import torch
    has_pytorch = True
except:
    has_pytorch = False


class AttMAML(MAML):

    def _get_attention_param_indices(self):
        """Helper method to identify attention parameter indices in variables list.
        
        Returns
        -------
        tuple: (attention_start_idx, attention_end_idx)
        """
        learner = self.learner
        ifm_param_count = len(list(learner.ifm.parameters()))
        attention_param_count = len(list(learner.attention.parameters()))
        
        attention_start_idx = ifm_param_count
        attention_end_idx = ifm_param_count + attention_param_count
        
        return attention_start_idx, attention_end_idx
    
    def fit(self,
            steps: int,
            max_checkpoints_to_keep: int = 5,
            checkpoint_interval: int = 600,
            restore: bool = False):
        """Perform meta-learning to train the model.
        
        Inner loop: Only attention parameters are updated
        Outer loop: All parameters are updated
        
        Parameters
        ----------
        steps: int
            the number of steps of meta-learning to perform
        max_checkpoints_to_keep: int
            the maximum number of checkpoint files to keep.  When this number is reached, older
            files are deleted.
        checkpoint_interval: int
            the time interval at which to save checkpoints, measured in seconds
        restore: bool
            if True, restore the model from the most recent checkpoint before training
            it further
        """
        if restore:
            self.restore()
        checkpoint_time: float = time.time()
    
        # Main optimization loop.
        learner = self.learner
        variables: OneOrMany[torch.Tensor] = learner.variables
        
        # Get attention parameter indices
        attention_start_idx, attention_end_idx = self._get_attention_param_indices()
        
        for i in range(steps):
            self._pytorch_optimizer.zero_grad()
            for j in range(self.meta_batch_size):
                learner.select_task()
                updated_variables: OneOrMany[torch.Tensor] = variables
                
                for k in range(self.optimization_steps):
                    loss, _ = self.learner.compute_model(
                        learner.get_batch(), updated_variables, True)
                    
                    gradients: Tuple[torch.Tensor, ...] = torch.autograd.grad(
                        loss,
                        updated_variables,
                        grad_outputs=torch.ones_like(loss),
                        create_graph=True,
                        retain_graph=True)
                    
                    # Inner loop: Only update attention parameters
                    updated_variables = [
                        v if (g is None or idx < attention_start_idx or idx >= attention_end_idx) 
                        else v - self.learning_rate * g
                        for idx, (v, g) in enumerate(zip(updated_variables, gradients))
                    ]
                
                # Outer loop evaluation on updated variables
                meta_loss, _ = self.learner.compute_model(
                    learner.get_batch(), updated_variables, True)
                
                # Outer loop: Compute gradients for all parameters
                meta_gradients: Tuple[torch.Tensor, ...] = torch.autograd.grad(
                    meta_loss,
                    variables,
                    grad_outputs=torch.ones_like(meta_loss),
                    retain_graph=True)
                
                if j == 0:
                    summed_gradients: Union[Tuple[torch.Tensor, ...],
                                            List[torch.Tensor]] = meta_gradients
                else:
                    summed_gradients = [
                        s + g for s, g in zip(summed_gradients, meta_gradients)
                    ]
            
            # Apply accumulated gradients to all parameters
            ind: int = 0
            for param in self.learner.parameters():
                param.grad = summed_gradients[ind]
                ind = ind + 1
    
            self._pytorch_optimizer.step()
            if self._lr_schedule is not None:
                self._lr_schedule.step()
    
            # Do checkpointing.
            if i == steps - 1 or time.time() >= checkpoint_time + checkpoint_interval:
                self.save_checkpoint(max_checkpoints_to_keep)
                checkpoint_time = time.time()
    
    def train_on_current_task(self,
                              optimization_steps: int = 1,
                              restore: bool = True):
        """Perform a few steps of gradient descent to fine tune the model on the current task.
        
        Only attention parameters are updated (following inner loop protocol).
        
        Parameters
        ----------
        optimization_steps: int
            the number of steps of gradient descent to perform
        restore: bool
            if True, restore the model from the most recent checkpoint before optimizing
        """
        if restore:
            self.restore()
        
        variables: OneOrMany[torch.Tensor] = self.learner.variables
        
        # Get attention parameter indices
        attention_start_idx, attention_end_idx = self._get_attention_param_indices()
        
        # Create list to track which parameters should be updated (only attention parameters)
        for idx, param in enumerate(self.learner.parameters()):
            if attention_start_idx <= idx < attention_end_idx:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Create optimizer only for attention parameters
        attention_params = [param for idx, param in enumerate(self.learner.parameters()) 
                           if attention_start_idx <= idx < attention_end_idx]
        
        task_optimizer: Optimizer = GradientDescent(
            learning_rate=self.learning_rate)
        
        pytorch_task_optimizer = task_optimizer._create_pytorch_optimizer(
            attention_params)
        if isinstance(task_optimizer.learning_rate, LearningRateSchedule):
            lr_schedule = task_optimizer.learning_rate._create_pytorch_schedule(
                pytorch_task_optimizer)
        else:
            lr_schedule = None
        
        for i in range(optimization_steps):
            pytorch_task_optimizer.zero_grad()
            inputs = self.learner.get_batch()
            loss, _ = self.learner.compute_model(inputs, variables, True)
            loss.backward()
            pytorch_task_optimizer.step()
        
        # Restore requires_grad for all parameters
        for param in self.learner.parameters():
            param.requires_grad = True
