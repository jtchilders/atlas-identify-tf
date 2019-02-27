import os,argparse,logging,json,glob,time
import numpy as np
import tensorflow as tf
import data_handling_calo2d as dh
import model_yolo_calo2d as model
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
   parser.add_argument('--batch_queue_size',type=int,default=4,
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
   rank = 0
   nranks = 1
   if args.horovod:
      import horovod.tensorflow as hvd
      hvd.init()
      rank = hvd.rank()
      nranks = hvd.size()
      if rank > 0:
         log_level = logging.ERROR
      tf.logging.set_verbosity(log_level)
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:' + '{:05d}'.format(rank) + ':%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('horovod from: %s',hvd.__file__)
      logger.info("Rank: %s of %s",rank,nranks)
   else:
      logging.basicConfig(level=log_level,format='%(asctime)s %(levelname)s:%(name)s:%(process)s:%(thread)s:%(message)s')
      logger.info('no MPI run')

   # do not shift random seed by rank
   # Batch Generator serves different files to different ranks at each batch
   # args.random_seed += rank

   logger.info('tensorflow from:       %s',tf.__file__)
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

   batches_per_epoch = int(float(len(trainlist)) / float(config_file['data_handling']['evt_per_file']))
   if args.batches_per_epoch != -1:
      batches_per_epoch = args.batches_per_epoch

   batches_per_epoch = int(batches_per_epoch / nranks)

   logger.info('creating datasets')
   # create datasets from the filelists
   if args.dstest:
      ds_train = dh.get_dummy_dataset(trainlist,batch_size,args.ds_procs,repeat=100,config_file=config_file)
      ds_valid = dh.get_dummy_dataset(validlist,batch_size,args.ds_procs,repeat=100,config_file=config_file)
   else:
      ds_train = dh.get_dataset(trainlist,batch_size,args.ds_procs)
      ds_valid = dh.get_dataset(validlist,batch_size,args.ds_procs)

   if args.horovod:
      logger.info('create shards')
      ds_train = ds_train.shard(nranks,rank)
      ds_valid = ds_valid.shard(nranks,rank)

   logger.info('creating iterators')
   # get iterators for the datasets
   handle,next_batch,iter_train,iter_valid = dh.get_iterators(ds_train,ds_valid,config_file)

   # the next_batch represents the (image,truth)
   input_image,truth = next_batch

   # get the model to run
   logger.info('creating model')
   prediction,grid_h,grid_w = model.build_model(input_image,1,len(config_file['data_handling']['classes']),config_file['training']['batch_size'])

   # create a loss function and apply to truth/prediction
   logger.info('creating lossop')
   lossop = loss.loss(truth,prediction)
   tf.summary.scalar("loss", lossop)

   p1 = tf.print('rank:',rank,'truth:',truth)
   p2 = tf.print('rank:',rank,'prediction:',prediction)
   with tf.control_dependencies([p1,p2]):
      lossop = tf.identity(lossop)

   # create an optimizer and minimize loss
   logger.info('creating optimizer')
   opt = tf.train.GradientDescentOptimizer(config_file['training']['learning_rate'])

   # create a global step count
   global_step = tf.train.get_or_create_global_step()

   # if using horovod
   hooks = []
   if args.horovod:
      logger.info('adding horovod optimizer')
      opt = hvd.DistributedOptimizer(opt)

      # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
      # from rank 0 to all other processes. This is necessary to ensure consistent
      # initialization of all workers when training is started with random weights
      # or restored from a checkpoint.
      hooks.append(hvd.broadcast_global_variables(0))

   logger.info('add compute gradient opt')
   # Compute the gradients for a list of variables.
   grads_and_vars = opt.compute_gradients(lossop, tf.trainable_variables())
   # Ask the optimizer to apply the capped gradients.
   train = opt.apply_gradients(grads_and_vars,global_step=global_step)

   # need an accuracy
   
   accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.round(prediction),tf.cast(truth,tf.float32)),tf.float32),name='accuracy')
   tf.summary.scalar('accuracy',accuracy)

   # print the trainable parameters for reference
   print_trainable_pars()

   # Create summaries to visualize weights
   logger.info('adding trainable pars to histograms')
   for var in tf.trainable_variables():
      tf.summary.histogram(var.name.replace(':','_'), var)
   # Summarize all gradients
   for grad, var in grads_and_vars:
      if grad is not None:
         tf.summary.histogram(var.name.replace(':','_') + '/gradient', grad)

   # op to merge all the summaries for recording (must run in session)
   logger.info('merging summary data opt')
   merged_summary_op = tf.summary.merge_all()

   # op to write logs to Tensorboard
   logger.info('create summary writer')
   if rank == 0:
      summary_writer = tf.summary.FileWriter(args.tb_logdir, graph=tf.get_default_graph())


   # Session begins here.

   # create the session for running
   logger.info('start session')
   config_proto = create_config_proto(args)
   sess = tf.Session(config=config_proto)

   # create the handles for each dataset for running
   logger.info('create iterator handles')
   handle_train = sess.run(iter_train.string_handle())
   # handle_valid = sess.run(iter_valid.string_handle())

   # initialize all the variables
   logger.info('init globals')
   init = tf.global_variables_initializer()
   sess.run(init)

   logger.info('run hooks')
   for hook in hooks:
      logger.info('running hook: "%s" ',hook)
      sess.run(hook)

   sum_duration = 0.
   sum2_duration = 0.
   n = 0.
   for epoch_num in range(num_epochs):

      logger.info('   >>>>>>>>> epoch %s <<<<<<<<<',epoch_num)

      batch_num = 0
      while True:
         try:
            start = time.time()
            logger.info('batch: %10d of %10d',batch_num,batches_per_epoch)

            _,loss_val,summary,acc_val = sess.run([train,lossop,merged_summary_op,accuracy],feed_dict = { handle: handle_train })
         
            # Write logs at every iteration
            if rank == 0:
               summary_writer.add_summary(summary, epoch_num*batches_per_epoch + batch_num)
            batch_num += 1

            duration = time.time() - start
            sum_duration += duration
            sum2_duration += duration*duration
            n += 1.
            time_to_completion = (batches_per_epoch - batch_num) * (sum_duration / n)
            logger.info(' loss = %10.4f; acurracy = %10.5f; duration = %10.5f; completed in = %10.5f',loss_val,accuracy_val,time.time() - start,(time_to_completion)/60.)

         except tf.errors.OutOfRangeError:
            logger.info('   <<<<<<<<< epoch %s >>>>>>>>>',epoch_num)
            break


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

def create_config_proto(params):
   '''EJ: TF config setup'''
   config = tf.ConfigProto()
   config.intra_op_parallelism_threads = params.num_intra
   config.inter_op_parallelism_threads = params.num_inter
   config.allow_soft_placement         = True
   os.environ['KMP_BLOCKTIME'] = str(params.kmp_blocktime)
   os.environ['KMP_AFFINITY'] = params.kmp_affinity
   return config


if __name__ == "__main__":
   print('start main')
   main()

