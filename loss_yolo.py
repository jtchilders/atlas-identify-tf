import logging
import tensorflow as tf
import numpy as np
logger = logging.getLogger(__name__)


config_file = None
grid_w = None
grid_h = None
batch_size = None
class_wt = None 

coord_scale = 1.
no_object_scale = 1.
object_scale = 1.
class_scale = 1.

def set_config(config_file_in):
   global config_file,grid_w,grid_h,batch_size,class_wt
   config_file = config_file_in
   grid_w      = config_file['data_handling']['grid_w']
   grid_h      = config_file['data_handling']['grid_h']
   batch_size  = config_file['training']['batch_size']
   class_wt    = np.ones(len(config_file['data_handling']['classes']), dtype='float32')

def loss(y_true,y_pred):


   # truth/prediction shape: [8,180,7]
   # last index breakdown: [1/0,X,Y,W,H,jet,electron]
   # X/Y are in units of grids so if the object is inside 
   # unit grid height=1,width=100, then the x,y will be between 1. < 2.
   # and width will be between 100. < 101.
   # W/H are in units of grid cells as well so if the object is the exact
   # size of a grid cell, W/H = 1./1.



   ### Taken from YOLOv2 github
   """
   Adjust prediction
   """
   # create a dummy grid cell
   cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1)))
   # from Rui Wang:
   cell_y = tf.to_float(tf.reshape(tf.tile(tf.range(grid_h), [grid_w]), (1, grid_w, grid_h, 1)))
   cell_y = tf.transpose(cell_y, (0,2,1,3))
   cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, 1])

   ### adjust x and y 
   pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

   ### adjust w and h
   pred_box_wh = tf.exp(y_pred[..., 2:4]) # * np.reshape(self.anchors, [1,1,1,self.nb_box,2])

   ### adjust confidence
   pred_box_conf = tf.sigmoid(y_pred[..., 4])

   ### adjust class probabilities
   pred_box_class = y_pred[..., 5:]


   """
   Adjust ground truth
   """
   ### adjust x and y
   true_box_xy = y_true[..., :2] # relative position to the containing cell

   ### adjust w and h
   true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically

   ### adjust confidence
   true_wh_half = true_box_wh / 2.
   true_mins    = true_box_xy - true_wh_half
   true_maxes   = true_box_xy + true_wh_half

   pred_wh_half = pred_box_wh / 2.
   pred_mins    = pred_box_xy - pred_wh_half
   pred_maxes   = pred_box_xy + pred_wh_half

   intersect_mins  = tf.maximum(pred_mins,  true_mins)
   intersect_maxes = tf.minimum(pred_maxes, true_maxes)
   intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
   intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

   true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
   pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

   union_areas = pred_areas + true_areas - intersect_areas
   iou_scores  = tf.truediv(intersect_areas, union_areas)

   true_box_conf = iou_scores * y_true[..., 4]

   ### adjust class probabilities
   true_box_class = tf.argmax(y_true[..., 5:], -1)


   """
   Determine the masks
   """
   ### coordinate mask: simply the position of the ground truth boxes (the predictors)
   coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * coord_scale

   ### confidence mask: penelize predictors + penalize boxes with low IOU
   # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
   # true_xy = self.true_boxes[..., 0:2]
   # true_wh = self.true_boxes[..., 2:4]

   # true_wh_half = true_wh / 2.
   # true_mins    = true_xy - true_wh_half
   # true_maxes   = true_xy + true_wh_half

   # pred_xy = tf.expand_dims(pred_box_xy, 4)
   # pred_wh = tf.expand_dims(pred_box_wh, 4)

   # pred_wh_half = pred_wh / 2.
   # pred_mins    = pred_xy - pred_wh_half
   # pred_maxes   = pred_xy + pred_wh_half    

   # intersect_mins  = tf.maximum(pred_mins,  true_mins)
   # intersect_maxes = tf.minimum(pred_maxes, true_maxes)
   # intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
   # intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

   # true_areas = true_wh[..., 0] * true_wh[..., 1]
   # pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

   # union_areas = pred_areas + true_areas - intersect_areas
   # iou_scores  = tf.truediv(intersect_areas, union_areas)

   # best_ious = tf.reduce_max(iou_scores, axis=4)
   # conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * no_object_scale



   # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
   conf_mask = y_true[..., 4] * object_scale

   ### class mask: simply the position of the ground truth boxes (the predictors)
   class_mask = y_true[..., 4] * tf.gather(class_wt, true_box_class) * class_scale


   ## YOLO V2 has warm up lambda here ####

   """
   Finalize the loss
   """
   nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
   nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
   nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))

   loss_xy    = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
   loss_wh    = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
   loss_conf  = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
   loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
   loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

   loss = loss_xy + loss_wh + loss_conf + loss_class


   return loss


