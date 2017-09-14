from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from abc import ABCMeta
import copy
import numpy as np
from six.moves import xrange
import tensorflow as tf
import warnings
import logging

import cleverhans.utils as utils
import cleverhans.utils_tf as utils_tf
from cleverhans.attacks import Attack
from cleverhans.model import Model, CallableModelWrapper

import logging

_logger = utils.create_logger("cleverhans.attacks")


def gradient_graph(predictions, x, nb_candidate):
    """
    Create the gradient graph to be ran later in a TF session
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param x: The input placeholder
    :param nb_candidate: The number of classes to test against, i.e.,
                        deepfool only consider nb_candidate classes when
                        attacking (thus accelerate speed)
    :return: Gradient graph
    """
    # This function will return a list of TF gradients
    list_derivatives = []
    # Define the TF graph elements to compute our derivatives for each class
    for class_ind in xrange(nb_candidate):
        derivatives, = tf.gradients(predictions[:, class_ind], x)
        list_derivatives.append(derivatives)

    return tf.stack(list_derivatives, axis=1)


def deepfool_batch(sess, x, pred, logits, grads, X, nb_candidate, overshoot,
                   max_iter, clip_min, clip_max, nb_classes, feed=None):
    """
    Applies DeepFool to a batch of inputs
    :param sess: TF session
    :param x: The input placeholder
    :param pred: The model's sorted symbolic output of logits, only the top
                nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param X: Numpy array with sample inputs
    :param nb_candidate: The number of classes to test against, i.e.,
                        deepfool only consider nb_candidate classes when
                        attacking (thus accelerate speed)
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :param nb_classes: Number of model output classes
    :return: Adversarial examples
    """
    X_adv = np.zeros(X.shape)
    X_adv = deepfool_attack(sess, x, pred, logits, grads, X, nb_candidate,
                            overshoot, max_iter, clip_min, clip_max, feed=feed)

    return np.asarray(X_adv, dtype=np.float32)


def deepfool_attack(sess, x, predictions, logits, grads, sample, nb_candidate,
                    overshoot, max_iter, clip_min, clip_max, feed=None):
    """
    TensorFlow implementation of DeepFool.
    Paper link: see https://arxiv.org/pdf/1511.04599.pdf
    :param sess: TF session
    :param x: The input placeholder
    :param predictions: The model's sorted symbolic output of logits, only the
                       top nb_candidate classes are contained
    :param logits: The model's unnormalized output tensor (the input to
                   the softmax layer)
    :param grads: Symbolic gradients of the top nb_candidate classes, procuded
                 from gradient_graph
    :param sample: Numpy array with sample input
    :param nb_candidate: The number of classes to test against, i.e.,
                        deepfool only consider nb_candidate classes when
                        attacking (thus accelerate speed)
    :param overshoot: A termination criterion to prevent vanishing updates
    :param max_iter: Maximum number of iteration for DeepFool
    :param clip_min: Minimum value for components of the example returned
    :param clip_max: Maximum value for components of the example returned
    :return: an adversarial sample
    """
    import copy

    adv_x = copy.copy(sample)
    # Initialize the loop variables
    iteration = 0
    current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
    if current.shape == ():
        current = np.array([current])
    w = np.squeeze(np.zeros(sample.shape[1:4]))  # same shape as original image
    r_tot = np.zeros(sample.shape)
    original = current  # use original label as the reference

    _logger.debug("Starting DeepFool attack up to {} iterations".
                  format(max_iter))
    # Repeat this main loop until we have achieved misclassification
    while (np.any(current == original) and iteration < max_iter):

        if iteration % 5 == 0 and iteration > 0:
            _logger.info("Attack result at iteration {} is {}".format(
                iteration,
                current))
        gradients = sess.run(grads, feed_dict={x: adv_x})
        predictions_val = sess.run(predictions, feed_dict={x: adv_x})
        for idx in range(sample.shape[0]):
            pert = np.inf
            if current[idx] != original[idx]:
                continue
            for k in range(1, nb_candidate):
                w_k = gradients[idx, k, ...] - gradients[idx, 0, ...]
                f_k = predictions_val[idx, k] - predictions_val[idx, 0]
                # adding value 0.00001 to prevent f_k = 0
                pert_k = (abs(f_k) + 0.00001) / np.linalg.norm(w_k.flatten())
                if pert_k < pert:
                    pert = pert_k
                    w = w_k
            r_i = pert*w/np.linalg.norm(w)
            r_tot[idx, ...] = r_tot[idx, ...] + r_i

        adv_x = np.clip(r_tot + sample, clip_min, clip_max)
        current = utils_tf.model_argmax(sess, x, logits, adv_x, feed=feed)
        if current.shape == ():
            current = np.array([current])
        # Update loop variables
        iteration = iteration + 1

    # need more revision, including info like how many succeed
    _logger.info("Attack result at iteration {} is {}".format(iteration,
                 current))
    _logger.info("{} out of {}".format(sum(current != original),
                                       sample.shape[0]) +
                 " becomes adversarial examples at iteration {}".format(
                     iteration))
    # need to clip this image into the given range
    adv_x = np.clip((1+overshoot)*r_tot + sample, clip_min, clip_max)
    return adv_x



class DeepFool(Attack):

    """
    DeepFool is an untargeted & iterative attack which aims at finding the
    minimum adversarial perturbations in deep networks. The implementation
    here is w.r.t. the L2 norm.
    Paper link: "https://arxiv.org/pdf/1511.04599.pdf"
    """

    def __init__(self, model, back='tf', sess=None):
        """
        Create a DeepFool instance.
        """
        super(DeepFool, self).__init__(model, back, sess)

        if self.back == 'th':
            raise NotImplementedError('Theano version not implemented.')

        import tensorflow as tf
        self.structural_kwargs = ['over_shoot', 'max_iter',
                                  'clip_max', 'clip_min',
                                  'nb_candidate', 'nb_classes']

        if not isinstance(self.model, Model):
            self.model = CallableModelWrapper(self.model, 'logits')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param nb_candidate: The number of classes to test against, i.e.,
                            deepfool only consider nb_candidate classes when
                            attacking (thus accelerate speed)
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """

        import tensorflow as tf

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        assert self.nb_candidate <= self.nb_classes,\
            'nb_candidate should not be greater than nb_classes'

        # Define graph wrt to this input placeholder
        logits = self.model.get_logits(x)
        preds = tf.reshape(tf.nn.top_k(logits, k=self.nb_candidate)[0],
                           [-1, self.nb_candidate])
        grads = gradient_graph(preds, x, self.nb_candidate)

        # Define graph
        def deepfool_wrap(x_val):
            return deepfool_batch(self.sess, x, preds, logits, grads, x_val,
                                  self.nb_candidate, self.overshoot,
                                  self.max_iter, self.clip_min, self.clip_max,
                                  self.nb_classes)
        return tf.py_func(deepfool_wrap, [x], tf.float32)

    def parse_params(self, nb_candidate=10, overshoot=0.02, max_iter=50,
                     nb_classes=1001, clip_min=0., clip_max=1., **kwargs):
        """
        :param nb_candidate: The number of classes to test against, i.e.,
                            deepfool only consider nb_candidate classes when
                            attacking (thus accelerate speed)
        :param overshoot: A termination criterion to prevent vanishing updates
        :param max_iter: Maximum number of iteration for deepfool
        :param nb_classes: The number of model output classes
        :param clip_min: Minimum component value for clipping
        :param clip_max: Maximum component value for clipping
        """
        self.nb_candidate = nb_candidate
        self.overshoot = overshoot
        self.max_iter = max_iter
        self.nb_classes = nb_classes
        self.clip_min = clip_min
        self.clip_max = clip_max

        return True
