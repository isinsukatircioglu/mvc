
import os
import os.path
import logging
import time
from collections import defaultdict

import numpy as np

from . import utils
from . import summary

logger = logging.getLogger(__name__)

def managed_object(name, obj, filename_template="{name}_{{iter:06d}}.{extension}"):
    
    if hasattr(obj, "load_state_dict") and hasattr(obj, "state_dict"):
        # Torch module
        import torch
        return (obj, 
                filename_template.format(name=name, extension="pth"),
                lambda obj, filename: torch.save(obj.state_dict(), filename),
                lambda obj, filename: obj.load_state_dict(torch.load(filename)))
    
    raise ValueError("Unknown object type {}".format(obj))

def managed_objects(objects, filename_template="{name}_{{iter:06d}}.{extension}"):
    
    return [managed_object(k, v, filename_template) for k, v in objects.items()]

class Trainer(object):
    
    def __init__(self,
                 training_step,
                 save_every=None,
                 save_path=None,
                 managed_objects=None, # List of tuples (object, filename_template, save_function, load_function)
                 test_function=None,
                 test_every=None):
        
        self.training_step = training_step
        self.iter = 0
        
        self.save_path = save_path
        
        # Summary for registering data
        self.summary = summary.Summary()
        
        self.save_every = save_every
        self.managed_objects = managed_objects or []
        self.saved_at = set()
        
        # Add the summary to the list of managed objects.
        self.managed_objects.append((self.summary, "summary_{iter:06d}.h5", 
                                     lambda obj, filename: summary.save_h5(obj, filename),
                                     lambda obj, filename: summary.load_h5(obj, filename)))
        
        self.test_function = test_function
        self.test_every = test_every
    
    def save(self):
        
        if self.save_path is None:
            return
        
        if self.iter in self.saved_at:
            # Do not save same model twice
            return
        
        utils.makedirs(self.save_path)
        
        for obj, filename_template, save_function, _ in self.managed_objects:
            filename = filename_template.format(iter=self.iter)
            filename = os.path.join(self.save_path, filename)
            
            logger.info("Saving '{}'...".format(filename))
            save_function(obj, filename)
        
        self.saved_at.add(self.iter)
    
    def load(self, niter):
        
        if self.save_path is None:
            raise ValueError("`save_path` not set; cannot load a previous state")
        
        for obj, filename_template, _, load_function in self.managed_objects:
            filename = filename_template.format(iter=niter)
            filename = os.path.join(self.save_path, filename)
            
            logger.info("Loading '{}'...".format(filename))
            load_function(obj, filename)
        
        self.iter = niter
    
    def run_validation(self):
        
        if self.test_function is None:
            return
        
        logger.info("Validating network at iteration {}...".format(self.iter))
        
        test_output = self.test_function(self.iter)
        
        if "loss" in test_output:
            logger.info("\tValidation loss: {}".format(test_output["loss"]))
        
        # Register validation results
        for k, v in test_output.items():
            self.summary.register("testing." + k, self.iter, v)
    
    def train(self, num_iters, print_every=0, maxtime=np.inf):
        
        tic = time.clock()
        time_elapsed = 0
        
        # print_epoch_every = max(print_every // self.iters_per_epoch, 1)
        
        for _ in range(num_iters):
            
            step_output = self.training_step(self.iter)
            
            time_elapsed = time.clock() - tic
            
            # Register data
            loss = step_output["loss"]
            for k, v in step_output.items():
                self.summary.register("training." + k, self.iter, v)
            
            self.summary.register("training.time", self.iter, time_elapsed)
            
            # Check for nan
            if np.any(np.isnan(loss)):
                logger.error("Last loss is nan! Training diverged!")
                break
            
            # logger.info(info)
            if print_every and self.iter % print_every == 0:
                logger.info("Iteration {}...".format(self.iter))
                logger.info("\tTraining loss: {}".format(loss))
                logger.info("\tTime elapsed: {:.2f}s".format(time_elapsed))
            
            self.iter += 1
            
            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break
            
            # Validation
            if self.test_every and self.iter % self.test_every == 0:
                self.run_validation()
            
            # Save model and solver
            if self.save_every and self.iter % self.save_every == 0:
                self.save()
    
