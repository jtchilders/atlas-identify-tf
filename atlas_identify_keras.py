import os,argparse,logging,json,glob,time,sys
import numpy as np
import tensorflow as tf
import data_handling_calo2d_h5 as dh
import model_yolo_calo2d_keras as model
import loss_yolo as loss
logger = logging.getLogger(__name__)


def main():
   global rates
   ''' simple starter program that can be copied for use when starting a new script. '''
   parser = argparse.ArgumentParser(description='Atlas Training')
   parser.add_argument('--config_file', '-c',
                       help='configuration in standard json format.')
   parser.add_argument('--tb_logdir',
                       help='tensorboard logdir for this job.',default=None)
   parser.add_argument('--model_dir',
                       help='where to save model weights for this job.',default='')
   parser.add_argument('--horovod', default=False,
                       help='use Horovod',action='store_true')
   parser.add_argument('--myopt', default=False,
                       help='use my custom optimizer',action='store_true')
   parser.add_argument('--ml_comm', default=False,
                       help='use Cray PE ML Plugin',action='store_true')
   parser.add_argument('--batches_per_epoch','-b', default=-1, type=int,
                       help='limit the number of batches for training. default is all')
   parser.add_argument('--num_files','-n', default=-1, type=int,
                       help='limit the number of files to process. default is all')
   parser.add_argument('--ds_procs', default=1, type=int,
                       help='number of dataset files to read concurrently. Can cause memory explosion.')
   parser.add_argument('--num_intra', type=int,default=4,
                       help='num_intra')
   parser.add_argument('--num_inter', type=int,default=1,
                       help='num_inter')
   parser.add_argument('--kmp_blocktime', type=int, default=10,
                       help='KMP BLOCKTIME')
   parser.add_argument('--kmp_affinity', default='granularity=fine,compact,1,0',
                       help='KMP AFFINITY')
   parser.add_argument('--batch_queue_size',type=str,default='4',
                       help='number of batch queues in the fit_generator')
   parser.add_argument('--batch_queue_workers',type=int,default=0,
                       help='number of batch workers in the fit_generator')
   parser.add_argument('--timeline',action='store_true',default=False,
                       help='enable chrome timeline profiling')
   parser.add_argument('--adam',action='store_true',default=False,
                       help='use adam optimizer')
   parser.add_argument('--sgd',action='store_true',default=False,
                       help='use SGD optimizer')
   parser.add_argument('--timeline_filename',default='timeline_profile.json',
                       help='filename to use for timeline json data')
   parser.add_argument('--random-seed', type=int,default=0,dest='random_seed',
                       help="Set the random number seed. This needs to be the same for all ranks to ensure the batch generator serves data properly.")

   parser.add_argument('--dstest',action='store_true',default=False,
                       help='use dummy dataset')


   args = parser.parse_args()

   print('loading MPI bits')
   log_level = logging.INFO
   fit_verbose = 1
   rank = 0
   nranks = 1
   if args.horovod:
      import horovod.tensorflow.keras as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         log_level = logging.WARNING
         fit_verbose = 0
      tf.logging.set_verbosity(tf.logging.WARN)
      os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1' # set Tensorflow C++ log level (0=INFO,1=WARN,2=ERROR,3=FATAL)
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('horovod from: %s',hvd.__file__)
      logger.info("Rank: %s of %s",rank,nranks)
   else:
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('no MPI run')

   config_proto = create_config_proto(args)
   tf.keras.backend.set_session(tf.Session(config=config_proto))

   # do not shift random seed by rank
   # Batch Generator serves different files to different ranks at each batch
   # args.random_seed += rank

   logger.info('tensorflow from:       %s',tf.__file__)
   logger.info('tensorflow version:    %s',tf.__version__)

   logger.info('tf.keras version:      %s',tf.keras.__version__)

   logger.info('config_file:           %s',args.config_file)
   logger.info('tb_logdir:             %s',args.tb_logdir)
   logger.info('model_dir:             %s',args.model_dir)
   logger.info('horovod:               %s',args.horovod)
   logger.info('myopt:                 %s',args.myopt)
   logger.info('adam:                  %s',args.adam)
   logger.info('sgd:                   %s',args.sgd)
   logger.info('num_files:             %s',args.num_files)
   logger.info('batches_per_epoch:     %s',args.batches_per_epoch)
   logger.info('ds_procs:              %s',args.ds_procs)
   logger.info('num_intra:             %s',args.num_intra)
   logger.info('num_inter:             %s',args.num_inter)
   logger.info('kmp_blocktime:         %s',args.kmp_blocktime)
   logger.info('kmp_affinity:          %s',args.kmp_affinity)
   logger.info('batch_queue_size:      %s',args.batch_queue_size)
   logger.info('batch_queue_workers:   %s',args.batch_queue_workers)
   logger.info('timeline:              %s',args.timeline)
   logger.info('timeline_filename:     %s',args.timeline_filename)
   logger.info('random-seed:           %s',args.random_seed)
   logger.info('dstest:                %s',args.dstest)

   np.random.seed(args.random_seed)

   # load configuration
   config_file = json.load(open(args.config_file))
   batch_size = config_file['training']['batch_size']
   num_epochs = config_file['training']['epochs']
   config_file['rank'] = rank
   config_file['nranks'] = nranks
   config_file['random_seed'] = args.random_seed

   # convert glob to filelists for training and validation

   trainlist,validlist = get_filelist(config_file,args)

   train_total_images = len(trainlist) * config_file['data_handling']['evt_per_file']
   train_total_batches = int(train_total_images / batch_size / nranks)

   if args.batches_per_epoch != -1:
      train_total_batches = args.batches_per_epoch
   
   valid_total_images = len(validlist) * config_file['data_handling']['evt_per_file']
   valid_total_batches = int(valid_total_images / batch_size / nranks)


   # get the model to run
   logger.info('creating model')
   keras_model,grid_h,grid_w = model.build_model(config_file)

   config_file['data_handling']['grid_h'] = grid_h
   config_file['data_handling']['grid_w'] = grid_w

   # pass config file to the loss function
   loss.set_config(config_file)

   logger.info('creating datasets')
   # create datasets from the filelists
   if args.dstest:
      ds_train = dh.get_dummy_dataset(trainlist,batch_size,args.ds_procs,repeat=100,config_file=config_file)
      ds_valid = dh.get_dummy_dataset(validlist,batch_size,args.ds_procs,repeat=100,config_file=config_file)
   else:
      ds_train = dh.get_dataset(trainlist,batch_size,config_file,args.ds_procs)
      ds_valid = dh.get_dataset(validlist,batch_size,config_file,args.ds_procs)

   #logger.info('creating iterators')
   # get iterators for the datasets
   #handle,next_batch,iter_train,iter_valid = dh.get_iterators(ds_train,ds_valid,config_file)

   # the next_batch represents the (image,truth)
   #input_image,truth = next_batch

   

   # create an optimizer and minimize loss
   logger.info('creating optimizer')
   opt = tf.keras.optimizers.SGD(config_file['training']['learning_rate'])

   # if using horovod
   callbacks = None
   if args.horovod:
      logger.info('adding horovod optimizer')
      opt = hvd.DistributedOptimizer(opt)

      # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
      # from rank 0 to all other processes. This is necessary to ensure consistent
      # initialization of all workers when training is started with random weights
      # or restored from a checkpoint.
      callbacks = []
      #callbacks.append(hvd.broadcast_global_variables(0))
      callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

      if rank == 0:
         callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.tb_logdir,
                                  histogram_freq=1,
                                  write_graph=True,
                                  write_grads=False,
                                  write_images=False))

   keras_model.summary()
   keras_model.compile(
      optimizer=opt,
      loss=tf.keras.losses.mean_squared_error, #loss.loss,
      metrics=['accuracy']
      )

   keras_model.fit(ds_train,
      #batch_size=config_file['training']['batch_size'],
      epochs=config_file['training']['epochs'],
      steps_per_epoch=train_total_batches,
      callbacks=callbacks,
      validation_data=ds_valid,
      verbose=fit_verbose,
      validation_steps=10,
      )

   


def print_trainable_pars():
   total_parameters = 0
   for variable in tf.trainable_variables():
       # shape is an array of tf.Dimension
       shape = variable.get_shape()
       variable_parameters = 1
       for dim in shape:
           variable_parameters *= dim.value
       total_parameters += variable_parameters
   logger.info('total trainable parameters: %s',total_parameters)


def get_filelist(config_file,args):
   # get file list
   filelist = sorted(glob.glob(config_file['data_handling']['input_file_glob']))
   logger.info('found %s input files',len(filelist))
   if len(filelist) < 2:
      raise Exception('length of file list needs to be at least 2 to have train & validate samples')

   nfiles = len(filelist)
   if args.num_files > 0:
      nfiles = args.num_files

   
   train_file_index = int(config_file['data_handling']['training_to_validation_ratio'] * nfiles)
   first_file = filelist[0]
   np.random.shuffle(filelist)
   assert first_file != filelist[0]

   train_imgs = filelist[:train_file_index]
   valid_imgs = filelist[train_file_index:nfiles]
   logger.info('training index: %s',train_file_index)
   while len(valid_imgs) * config_file['data_handling']['evt_per_file'] / config_file['training']['batch_size'] < 1.:
      logger.info('training index: %s',train_file_index)
      train_file_index -= 1
      train_imgs = filelist[:train_file_index]
      valid_imgs = filelist[train_file_index:nfiles]

   logger.info('training files: %s; validation files: %s',len(train_imgs),len(valid_imgs))

   return train_imgs,valid_imgs

def create_config_proto(args):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = args.num_intra
   config.inter_op_parallelism_threads = args.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(args.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = str(args.kmp_affinity)
   return config


if __name__ == "__main__":
   print('start main')
   main()

