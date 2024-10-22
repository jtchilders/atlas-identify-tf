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
   feature_shape = (config_file['training']['batch_size'],) + tuple(config_file['data_handling']['image_shape'])
   label_shape = (config_file['training']['batch_size'],8,180,1,7)

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
   # start = time.time()
   npdata = np.load(filename)
   features = npdata['raw']  # (N, 16, 256, 5761)
   file_entries,channels,height,width = features.shape

   grid_w = 180
   grid_h = 8
   
   # reduce to just 2 channels, sum of EM and sum of HAD
   new_features = np.zeros([file_entries,2,height,int(width/2)*2],dtype=np.float32)
   # sum all EM layers
   new_features[:,0,:,:] = np.sum(features[:,8:12,:,:-1],axis=1)
   # sum all HAD layers
   new_features[:,1,:,:] = np.sum(features[:,12:16,:,:-1],axis=1)

   features = new_features
   logger.info('features = %s',features.shape)
   truth = npdata['truth']
   labels = np.zeros([file_entries,grid_h,grid_w,1,7],dtype=np.int32)

   for i in range(file_entries):
      for j in range(truth.shape[1]):
            gw = int(truth[i,j,1]/(width/grid_w))
            gh = int(truth[i,j,2]/(height/grid_h))
            labels[i,gh,gw,0,0:5] = truth[i,j,0:5]
            labels[i,gh,gw,0,5] = np.any(truth[i,j,5:10])
            labels[i,gh,gw,0,6] = np.any(truth[i,j,10:14])

   logger.info('truth = %s',labels.shape)
   #labels = labels[:,:,]
   #labels = labels[:,0,8]  # exctract b-jet only
   #labels = np.int32(labels[:,np.newaxis])
   # logger.debug('read filename %s in %10.2f' % (filename,time.time() - start))
   # logger.debug('shapes: features = %s; labels = %s',features.shape,labels.shape)
   return features,labels


# wrapper function to make this a representative function that Tensorflow
# can add to a graph
def get_data_wrapper(filename):
   features, labels = tf.py_func(func=get_data_from_filename,inp=[filename], Tout=(tf.float32, tf.int32))
   # return tf.data.Dataset.from_tensor_slices((features, labels))
   return (features,labels)


if __name__ == '__main__':
   from mpi4py import MPI
   logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(MPI.COMM_WORLD.Get_rank()) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
   import argparse,json,glob
   parser = argparse.ArgumentParser(description='testing dataset prep')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   parser.add_argument('--nimgs', '-n',
                       help='number of images to print.',default=10,type=int)
   args = parser.parse_args()

   config_file = json.load(open(args.config_file))
   batch_size = config_file['training']['batch_size']

   filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   logger.info('found %s input files',len(filelist))

   half_i = int(len(filelist)/2)
   trainlist = filelist[:half_i]
   validlist = filelist[half_i:]

   ds_train = get_dataset(trainlist,batch_size,10)
   ds_valid = get_dataset(validlist,batch_size,1)

   handle,next_batch,iter_train,iter_valid = get_iterators(ds_train,ds_valid,config_file)

   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = 64
   config.inter_op_parallelism_threads = 10
   config.allow_soft_placement         = True
   sess = tf.Session(config=config)

   handle_train = sess.run(iter_train.string_handle())

   for i in range(args.nimgs):
      logger.info('image: %s of %s',i,args.nimgs)
      features,labels = sess.run(next_batch,feed_dict={handle:handle_train})
      logger.info('shapes:  features =  %s; labels = %s',features.shape,labels.shape)
      mask = tf.greater(features,0.1)
      non_zero_features = tf.boolean_mask(features,mask)
      p1 = tf.print('non_zero_features:',non_zero_features,summarize=100)
      sess.run(p1)

      zero = tf.constant(0, dtype=tf.int32)
      where = tf.not_equal(labels[...,0], zero)
      indices = tf.where(where)

      #non_zero_labels = tf.boolean_mask(labels,mask)
      logger.info('labels: %s',sess.run(indices))
      






  