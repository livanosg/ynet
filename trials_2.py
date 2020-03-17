import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


# latest_ckp = tf.train.latest_checkpoint('/home/medphys/projects/ynet/saves/MRynet')
# print_tensors_in_checkpoint_file(latest_ckp,all_tensors=False, all_tensor_names=True, tensor_name='')
#
latest_ckp = tf.train.latest_checkpoint('/home/medphys/projects/ynet/saves/MR_trial_0')
print_tensors_in_checkpoint_file(latest_ckp,all_tensors=False, all_tensor_names=True, tensor_name='')