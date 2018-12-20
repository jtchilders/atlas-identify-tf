import time,logging
import tensorflow as tf
import numpy as np
logger = logging.getLogger(__name__)


def get_dummy_dataset(filelist,batch_size,num_parallel_calls=1,repeat=1000,config_file=None):

   feature_shape = [100] + config_file['data_handling']['image_shape'] + [1]
   features = np.float32(np.random.random(feature_shape))

   label_shape = [100,1]
   labels = np.int32(np.random.random(label_shape))
   
   ds = tf.data.Dataset.from_tensor_slices((features,labels))

   return ds.batch(batch_size).repeat(repeat)


def get_dataset(filelist,batch_size,num_parallel_calls=1,repeat=1):
   # create dataset of filenames
   ds = tf.data.Dataset.from_tensor_slices(filelist)
   # map those filenames into feature+label elements, using map to get parallel behavior
   ds = ds.map(get_data_wrapper,num_parallel_calls=num_parallel_calls)
   # now flatten dataset
   ds = ds.flat_map(lambda *x: tf.data.Dataset.from_tensor_slices(x))
   # set batch size
   ds = ds.batch(batch_size)
   # set repeat per epoch
   ds = ds.repeat(repeat)

   return ds


def get_iterators(ds_train,ds_valid,config_file):
   # A feedable iterator is defined by a handle placeholder and its structure. We
   # could use the `output_types` and `output_shapes` properties of either
   # `training_dataset` or `validation_dataset` here, because they have
   # identical structure.
   feature_shape = (config_file['training']['batch_size'],) + tuple(config_file['data_handling']['image_shape']) + (1,)
   label_shape = (config_file['training']['batch_size'],len(config_file['data_handling']['classes']))

   handle = tf.placeholder(tf.string, shape=[])
   iterator = tf.data.Iterator.from_string_handle(
       handle, (tf.float32,tf.int32), (feature_shape,label_shape))
   next_element = iterator.get_next()

   # You can use feedable iterators with a variety of different kinds of iterator
   # (such as one-shot and initializable iterators).
   iter_train = ds_train.make_one_shot_iterator()
   iter_valid = ds_valid.make_one_shot_iterator()

   # get the iterator function
   return handle,next_element,iter_train,iter_valid


# take a string filename and open the compressed numpy files
# return features and labels
def get_data_from_filename(filename):
   filename = filename.decode('utf-8')
   # logger.debug('reading filename %s' % filename)
   start = time.time()
   npdata = np.load(filename)
   features = npdata['raw']
   features = np.float32(features[:,:,:,:,np.newaxis])
   labels = npdata['truth']
   labels = labels[:,0,8]  # exctract b-jet only
   labels = np.int32(labels[:,np.newaxis])
   # logger.debug('read filename %s in %10.2f' % (filename,time.time() - start))
   # logger.debug('shapes: features = %s; labels = %s',features.shape,labels.shape)
   return features,labels


# wrapper function to make this a representative function that Tensorflow
# can add to a graph
def get_data_wrapper(filename):
   features, labels = tf.py_func(get_data_from_filename, [filename], (tf.float32, tf.int32))
   # return tf.data.Dataset.from_tensor_slices((features, labels))
   return (features,labels)

  