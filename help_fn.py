import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from config import paths


def get_tensors_names(path):
    """ Get tensors' names from checkpoint useful for initialize model from checkpoint through warm-start
        Args:
        path: The path to checkpoint.

        :return All tensors' names from checkpoint """
    latest_ckp = tf.train.latest_checkpoint(path)
    print_tensors_in_checkpoint_file(latest_ckp, all_tensors=False, all_tensor_names=True, tensor_name='')


def get_tensors_values(path, tensor_name=''):
    """ Inspect tensors' values from checkpoint
    Args:
        path: The path to checkpoint.
        tensor_name: If provided returns the values of this tensor otherwise returns all tensors' values.
    :return Tensors values
        """
    latest_ckp = tf.train.latest_checkpoint(path)
    if tensor_name:
        all_tensors = False
    else:
        all_tensors = True
    print_tensors_in_checkpoint_file(latest_ckp, tensor_name=tensor_name, all_tensors=all_tensors,
                                     all_tensor_names=False)


def cyclic_learning_rate(global_step, learning_rate=0.01, max_lr=0.1, step_size=20., gamma=0.99994, mode='triangular',
                         name=None):
    """https: // github.com / mhmoodlan / cyclic - learning - rate / blob / master / clr.py"""
    """Applies cyclic learning rate (CLR).
     From the paper: Smith, Leslie N. "Cyclical learning rates for training neural networks." 2017. 
     [https://arxiv.org/pdf/1506.01186.pdf]
      This method lets the learning rate cyclically vary between reasonable boundary values achieving improved 
      classification accuracy and often in fewer iterations. This code varies the learning rate linearly between the
     minimum (learning_rate) and the maximum (max_lr). It returns the cyclic learning rate. It is computed as:
       ```python
      cycle = floor( 1 + global_step / ( 2 * step_size ) )
      x = abs( global_step / step_size – 2 * cycle + 1 )
      clr = learning_rate + ( max_lr – learning_rate ) * max( 0 , 1 - x )
       ```Polices:
        'triangular': Default, linearly increasing then linearly decreasing the learning rate at each cycle.
        'triangular2': The same as the triangular policy except the learning rate difference is cut
        in half at the end of each cycle. This means the learning rate difference drops after each cycle.
        'exp_range': The learning rate varies between the minimum and maximum boundaries and each boundary value 
        declines by an exponential factor of: gamma^global_step.
       Args:
        global_step: A scalar `int32` or `int64` `Tensor` or a Python number.
        learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
                       The initial learning rate which is the lower bound of the cycle (default = 0.1).
        max_lr:  A scalar. The maximum learning rate boundary.
        step_size: A scalar. The number of iterations in half a cycle. The paper suggests step_size = 2-8 x training iterations in epoch.
        gamma: constant in 'exp_range' mode: gamma**(global_step)
        mode: one of {triangular, triangular2, exp_range}. Default 'triangular'. Values correspond to policies detailed above.
        name: String.  Optional name of the operation.  Defaults to 'CyclicLearningRate'.
       Returns:
        A scalar `Tensor` of the same type as `learning_rate`. The cyclic learning rate.
      Raises:
        ValueError: if `global_step` is not supplied.
      @compatibility(eager)
      When eager execution is enabled, this function returns a function which in turn returns the decayed learning
      rate Tensor. This can be useful for changing the learning rate value across different invocations of optimizer functions.
      @end_compatibility
  """
    if global_step is None:
        raise ValueError("global_step is required for cyclic_learning_rate.")
    with ops.name_scope(name, "CyclicLearningRate",
                        [learning_rate, global_step]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(step_size, dtype)

        def cyclic_lr():
            """Helper to recompute learning rate; most helpful in eager-mode."""
            # computing: cycle = floor( 1 + global_step / ( 2 * step_size ) )
            double_step = math_ops.multiply(2., step_size)
            global_div_double_step = math_ops.divide(global_step, double_step)
            counter = math_ops.add(1., global_div_double_step)
            cycle = math_ops.floor(counter)
            # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
            double_cycle = math_ops.multiply(2., cycle)
            global_div_step = math_ops.divide(global_step, step_size)
            tmp = math_ops.subtract(global_div_step, double_cycle)
            x = math_ops.abs(math_ops.add(1., tmp))
            # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
            a1 = math_ops.maximum(0., math_ops.subtract(1., x))
            a2 = math_ops.subtract(max_lr, learning_rate)
            clr = math_ops.multiply(a1, a2)
            # clr = tf.where(tf.equal(counter, cycle + 0.5), -clr, clr)

            if mode == 'triangular2':
                clr = math_ops.divide(clr,
                                      math_ops.cast(math_ops.pow(2, math_ops.cast(cycle - 1, tf.int32)), tf.float32))
            if mode == 'exp_range':
                clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
            return math_ops.add(clr, learning_rate, name=name)

        if not context.executing_eagerly():
            cyclic_lr = cyclic_lr()
        return cyclic_lr


def f1(labels, predictions):  # Macro average
    # [b, h*w, classes]
    class_freq = tf.reduce_sum(labels, axis=[0, 1, 2])
    numerator = tf.reduce_sum(labels * predictions, axis=[0, 1, 2])
    numerator = tf.where(tf.equal(class_freq, 0), tf.ones_like(numerator), numerator)
    denominator = tf.reduce_sum(labels + predictions, axis=[0, 1, 2])
    denominator = tf.where(tf.equal(class_freq, 0), tf.ones_like(denominator) * 2, denominator)
    dice = 2. * numerator / denominator
    return tf.compat.v1.metrics.mean(dice)


def get_model_paths(args):
    trial = 0
    while os.path.exists(paths['save'] + '/{}_trial_{}'.format(args.modality, trial)):
        trial += 1
    model_path = paths['save'] + '/{}_trial_{}'.format(args.modality, trial)
    eval_path = model_path + '/eval'
    return model_path, eval_path
