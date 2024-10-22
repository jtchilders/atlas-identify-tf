import time,logging
import tensorflow as tf
import numpy as np
import h5py
logger = logging.getLogger(__name__)


class filegen:
   def __init__(self,config_file):
      self.config_file = config_file
      shape = config_file['data_handling']['image_shape']
      self.img_width = shape[2]
      self.img_height = shape[1]
      self.channels = shape[0]
      self.grid_h = config_file['data_handling']['grid_h']
      self.grid_w = config_file['data_handling']['grid_w']
      self.batch_size = config_file['training']['batch_size']
      self.num_classes = len(config_file['data_handling']['classes'])
      self.num_boxes = config_file['model_pars']['num_boxes']
      self.img_shape = config_file['data_handling']['image_shape']
   def __call__(self,filename):
      logger.warning('opening filename: %s',filename)
      with h5py.File(filename,'r') as hf:
         raw_data = hf['raw']
         truth = self.convert_truth(hf['truth'])
         logger.warning('raw_data shape: %s  truth shape: %s',raw_data.shape,truth.shape)
         assert(len(raw_data) == len(truth))
         for i in range(len(raw_data)):
            try:
               yield raw_data[i,:,:,:-1],truth[i]
            except:
               logger.exception('failed to get data from file: %s, for index %s, raw_data shape: %s, truth shape: %s',filename,i,raw_data.shape,truth.shape)
               raise

   def convert_truth(self,truth):

      pix_per_grid_w = self.img_width / self.grid_w
      pix_per_grid_h = self.img_height / self.grid_h

      new_truth = np.zeros((len(truth),
                            self.grid_h,
                            self.grid_w,
                            4+1+self.num_classes))

      for img_num in range(len(truth)):
         img_truth = truth[img_num]
         for obj_num in range(len(img_truth)):
            obj_truth = img_truth[obj_num]

            obj_exists   = obj_truth[0]

            if obj_exists == 1:

               obj_center_x = obj_truth[1] / pix_per_grid_w
               obj_center_y = obj_truth[2] / pix_per_grid_h
               obj_width    = obj_truth[3] / pix_per_grid_w
               obj_height   = obj_truth[4] / pix_per_grid_h

               grid_x = int(np.floor(obj_center_x))
               grid_y = int(np.floor(obj_center_y))

               if grid_x >= self.grid_w:
                  raise Exception('grid_x %s is not less than grid_w %s' % (grid_x,self.grid_w))
               if grid_y >= self.grid_h:
                  raise Exception('grid_y %s is not less than grid_h %s' % (grid_y,self.grid_h))

               new_truth[img_num,grid_y,grid_x,0:5] = [obj_center_x,obj_center_y,obj_width,obj_height,1]
               new_truth[img_num,grid_y,grid_x,5] = np.sum(obj_truth[5:10])
               new_truth[img_num,grid_y,grid_x,6] = np.sum(obj_truth[10:12])

      return new_truth






def get_dataset(filelist,batch_size,config_file,num_parallel_calls=1,repeat=1):
   # create dataset of filenames
   ds = tf.data.Dataset.from_tensor_slices(filelist)

   # shard if necessary
   ds = ds.shard(config_file['nranks'],config_file['rank'])

   # shuffle
   ds = ds.shuffle(len(filelist))

   # create dataset from generator
   ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
            generator=filegen(config_file),
            output_types=(tf.float32,tf.int32),
            output_shapes=(tf.TensorShape(config_file['data_handling']['image_shape']),
                  tf.TensorShape([config_file['data_handling']['grid_h'],config_file['data_handling']['grid_w'],4+1+2])),
            args=(filename,)
               ),
         cycle_length=num_parallel_calls,
         block_length=batch_size,
         num_parallel_calls=num_parallel_calls
         )

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
   label_shape = (config_file['training']['batch_size'],5,14)

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


   config_file['data_handling']['grid_h'] = 8
   config_file['data_handling']['grid_w'] = 180

   ds_train = get_dataset(trainlist,batch_size,config_file,10)
   ds_valid = get_dataset(validlist,batch_size,config_file,1)

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
      #p1 = tf.print('non_zero_features:',non_zero_features,summarize=100)
      #sess.run(p1)

      zero = tf.constant(0, dtype=tf.int32)
      where = tf.not_equal(labels[...,0], zero)
      indices = tf.where(where)

      #non_zero_labels = tf.boolean_mask(labels,mask)
      logger.info('labels: %s',sess.run(indices))
      






  