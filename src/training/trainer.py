import os
import os.path
import logging
import time
import numpy as np
from glob import glob

import torch
import numbers
import IPython

from . import utils
from . import summary

logger = logging.getLogger(__name__)


def managed_object(name, obj, filename_template="{name}_{{iteration}}.{extension}"):
    return (obj,
            filename_template.format(name=name, extension="pth"),
            lambda some_obj, filename: torch.save(some_obj.state_dict(), filename),
            lambda some_obj, filename: utils.transfer_partial_weights(torch.load(filename), some_obj)
            )


def managed_objects(objects, filename_template="{name}_{{iteration}}.{extension}"):
    """
    :param objects: Dictionary where the key is the name and the value is the object
    :param filename_template:
    :return: List of tuples (object, filename_template, save_function, load_function)
    """
    return [managed_object(k, v, filename_template) for k, v in objects.items()]


class Trainer(object):
    def __init__(self,
                 training_step,
                 save_every=None,
                 save_path=None,
                 managed_objects=None,  # List of tuples (object, filename_template, save_function, load_function)
                 test_functions=None,
                 test_every=None,
                 plot_instance=None):
        self.training_step = training_step  # This is the actual training function
        self.iteration = 0
        self.save_path = save_path
        self.save_every = save_every
        self.val_save_objects = managed_objects or []
        self.managed_objects = managed_objects or []
        self.test_functions = test_functions
        self.test_every = test_every
        self.plot_instance = plot_instance
        if not self.plot_instance:
            print("No plot instance given")

        self.saved_at = set()
        self.summary = summary.Summary()
        # Add the summary to the list of managed objects.
        self.managed_objects.append((self.summary, "summary_{iteration}.h5",
                                     lambda obj, filename: summary.save_h5(obj, filename),
                                     lambda obj, filename: summary.load_h5(obj, filename)))

        self.best_loss = {}
        self.best_val_J = 0.
        self.best_val_F = 0.

    def construct_save_name(self, filename_template, iteration):
        if isinstance(iteration, numbers.Number):
            filename = filename_template.format(iteration='{:06d}'.format(iteration))
        else:
            filename = filename_template.format(iteration=iteration)
        return filename

    def save(self, iteration, only_model=False):
        if self.save_path is None:
            return

        if iteration is None:
            iteration = self.iteration

        model_save_folder = os.path.join(self.save_path, 'models')
        utils.makedirs(model_save_folder)

        saved_objs = self.val_save_objects if only_model else self.managed_objects

        for obj, filename_template, save_function, _ in saved_objs:
            filename = self.construct_save_name(filename_template, iteration)
            filename = os.path.join(model_save_folder, filename)
            logger.info("Saving '{}'...".format(filename))
            save_function(obj, filename)

        self.saved_at.add(self.iteration)

    def load(self, niter):
        if self.save_path is None:
            raise ValueError("`save_path` not set; cannot load a previous state")

        model_save_path = os.path.join(self.save_path, 'models')

        for obj, filename_template, _, load_function in self.managed_objects:
            filename = self.construct_save_name(filename_template, niter)
            filename = os.path.join(model_save_path, filename)

            logger.info("Loading '{}'...".format(filename))
            load_function(obj, filename)

        # setting the best val loss, to decide whether the optimized model is improving and should be stored
        self.iteration = self.summary.get('training.time')[0][-1]
        for k in range(0, 100):  # iterate over all losses and set the best val loss
            tag = 'best_val_loss_t{}'.format(k)
            if self.summary.has_tag(tag):
                self.best_loss[k] = self.summary.get(tag)[1][-1]
        logger.info("Loaded model and summary at iteration '{}', best_loss={}".format(self.iteration, self.best_loss))

    def run_validation(self):
        # validate on all validation sets and save the best model
        for ti, test_function in enumerate(self.test_functions):

            if self.iteration % 50 == 0 and self.iteration != 0 and ti == 0:
                logger.info("Checking J Score every 1000 iteration and saving the model...")
                logger.info("Save path: {}".format(self.save_path))
                test_output = test_function('validation_t{}_i{}'.format(ti, self.iteration))  # self.iteration,

                if test_output['best_J'] > self.best_val_J:

                    old_files = [x for x in os.listdir(self.save_path + '/models') if 'best_val_J' in x]

                    for f in old_files:
                        os.remove(self.save_path + '/models/' + f)

                    self.best_val_J = test_output['best_J']
                    self.best_val_F = test_output['best_F']
                    logger.info("\tSaving Model: Validation J Score: {:.3f} - F Score {:.3f} - Threshold: {} - Iteration: {}".format(
                        test_output['best_J'], self.best_val_F, test_output['best_J_th'], self.iteration))
                    self.save('best_val_J_{:.3f}_i{}'.format(self.best_val_J, self.iteration), True)
                else:
                    logger.info(
                        "\tNOT SAVING!: Validation J Score: {:.3f} is lower than current J score: {} - F Score {:.3f} - Iteration: {}".format(
                            test_output['best_J'], self.best_val_J, self.best_val_F, self.iteration))

            if self.iteration % self.test_every[
                ti] != 0:  # and self.iteration != 2000: # run test as specified and once after X iter
                continue

            logger.info("Validating network at iteration {}, test function {}...".format(self.iteration, ti))

            test_output = test_function('validation_t{}_i{}'.format(ti, self.iteration))  # self.iteration,

            # Register validation results
            for k, v in test_output.items():
                self.summary.register("validation.t{}.{}".format(ti, k), self.iteration, v)

            # Save the best loss so far
            for k, v in self.best_loss.items():
                self.summary.register('best_val_loss_t{}'.format(k), self.iteration, v)

            logger.info(
                "\tValidation J Score: {:.3f} - F Score {:.3f} - Threshold: {}".format(test_output['best_J'], test_output['best_F'], test_output['best_J_th']))

            if "loss" in test_output:
                logger.info("\tValidation loss: {}".format(test_output["loss"]))
                if (self.iteration > 0 and ti > 0) and (
                        ti not in self.best_loss or test_output["loss"] < self.best_loss[
                    ti]):  # ti > 0, don't save for training loss
                    self.best_loss[ti] = test_output["loss"]
                    self.save('best_val_t{}'.format(ti))

        # always save the latest version, to be able to continue training
        # done after validation, to store best_val_loss_t
        if any([self.iteration % self.test_every[ti] == 0 for ti in range(0, len(self.test_functions))]):
            self.save('last_val')

    def train(self, num_iters, print_every=0, maxtime=np.inf):
        tic = time.time()
        max_snapshots = 50
        if num_iters / self.save_every > max_snapshots:
            num_iters = max_snapshots * self.save_every
            print("WARNING, instructed to save more then {} snapshots, reducing number of iterations to {}".format(
                max_snapshots, num_iters))

        for some_iter in range(num_iters):
            step_output = self.training_step(self.iteration)

            time_elapsed = time.time() - tic

            # Register data
            loss = step_output["loss"]
            for k, v in step_output.items():
                self.summary.register("training." + k, self.iteration, v)

            self.summary.register("training.time", self.iteration, time_elapsed)
            # self.summary.register("training.iteration", self.iteration, self.iteration)

            # Check for nan
            if np.any(np.isnan(loss)):
                logger.error("Last loss is nan! Training diverged!")
                break

            # logger.info(info)
            if print_every and self.iteration % print_every == 0:
                logger.info("Iteration {} of {}...".format(self.iteration, num_iters))
                logger.info("\tTraining loss: {}".format(loss))
                logger.info("\tTime elapsed: {:.2f}s".format(time_elapsed))

            if time_elapsed > maxtime:
                logger.info("Maximum time reached!")
                break

            # Validation
            self.run_validation()

            # Save model and solver
            if self.save_every and self.iteration > 0 and self.iteration % self.save_every == 0:
                logger.info('Saving to ...')
                self.save(self.iteration)

            self.plot_instance.plot_train_info_iteration(self.summary, self.iteration, self.save_path)

            self.iteration += 1

