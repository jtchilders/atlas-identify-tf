import logging
import tensorflow as tf
import numpy as np
logger = logging.getLogger(__name__)

def loss(truth,prediction):

   logger.info('truth: %s; pred: %s',truth,prediction)
   # pred  shape: [N,grid_h,grid_w,num_boxes,4+1+num_classes]
   pred_shape = prediction.get_shape()
   batch_size = pred_shape[0].value
   grid_h = pred_shape[1].value
   grid_w = pred_shape[2].value
   num_classes = pred_shape[3].value - 5
   # truth shape: [N,num_particles,4+1+num_classes]



   





   return tf.losses.mean_squared_error(truth,prediction)


