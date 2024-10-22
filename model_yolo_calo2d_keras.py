import tensorflow as tf
from tensorflow.keras import layers,models
import logging,sys
logger = logging.getLogger(__name__)


layer_counter = 0



def build_model(config_file):
   global layer_counter

   num_labels = len(config_file['data_handling']['classes'])
   batch_size = config_file['training']['batch_size']
   num_boxes  = config_file['model_pars']['num_boxes']

   inputs = layers.Input(shape=tuple(config_file['data_handling']['image_shape']))
   logger.info('building model, input image: %s',inputs)
   

   # Layer 1
   outputs = FullLayer(inputs,32,(3,3),(2,2))

   # Layer 2
   outputs = FullLayer(outputs,64,(3,3),(2,2))

   # Layer 3
   outputs = PartialLayer(outputs,128,(3,3))
   
   # Layer 4
   outputs = PartialLayer(outputs,64,(1,1))

   # Layer 5
   outputs = FullLayer(outputs,128,(3,3),(2,2))
   
   # Layer 6
   outputs = PartialLayer(outputs,256,(3,3))

   # Layer 7
   outputs = PartialLayer(outputs,128,(1,1))

   # Layer 8
   outputs = FullLayer(outputs,256,(3,3),(2,2))
   
   # Layer 9
   outputs = PartialLayer(outputs,512,(3,3))

   # Layer 10
   outputs = PartialLayer(outputs,256,(1,1))

   # Layer 11
   outputs = PartialLayer(outputs,512,(3,3))

   # Layer 12
   outputs = PartialLayer(outputs,256,(1,1))

   # Layer 13
   outputs = FullLayer(outputs,512,(3,3),(2,2))

   #outputs = MaxPool2D((2,2))(outputs)


   connection = outputs


   # Layer 14
   outputs = PartialLayer(outputs,1024,(3,3))

   # Layer 15
   outputs = PartialLayer(outputs,512,(1,1))

   # Layer 16
   outputs = PartialLayer(outputs,1024,(3,3))

   # Layer 17
   outputs = PartialLayer(outputs,512,(1,1))

   # Layer 18
   outputs = PartialLayer(outputs,1024,(3,3))

   # Layer 19
   outputs = PartialLayer(outputs,1024,(3,3))

   # Layer 20
   outputs = PartialLayer(outputs,1024,(3,3))

   # Layer 21
   connection = PartialLayer(connection,64,(1,1))


   outputs = layers.Concatenate(axis=1)([connection,outputs])

   # Layer 22
   outputs = PartialLayer(outputs,1024, (3,3))

   grid_h = outputs.get_shape()[2].value
   grid_w = outputs.get_shape()[3].value

   logger.info('grid_h x grid_w =  %s x %s ',grid_h,grid_w)

   # make the object detection layer
   layer_counter += 1
   outputs = Conv2D(num_boxes * (4 + 1 + num_labels),
                  (1,1), strides=(1,1),
                  name='DetectionLayer_%s' % layer_counter,
                  kernel_initializer='lecun_normal')(outputs)
   
   layer_counter += 1
   outputs = layers.Reshape((grid_h, grid_w, 4 + 1 + num_labels),name='Reshape_%s'%layer_counter)(outputs)

   model = models.Model(inputs,outputs)

   
   return model,grid_h,grid_w


def add_summary(outputs,debug=True):
   return outputs
   tf.summary.histogram(outputs.name.replace(':','_'),outputs)
   logger.info('layer %s: %s',layer_counter,outputs)
   sys.stdout.flush()
   sys.stderr.flush()

   if debug:
      pr = tf.print('layer ',layer_counter)
      with tf.control_dependencies([pr]):
         outputs = tf.identity(outputs)

   return outputs



def FullLayer(inputs,filters,kernel_size=(3,3),pool_size=(2,2)):

   outputs = Conv2D(filters,kernel_size)(inputs)
   outputs = BatchNorm()(outputs)
   outputs = LeakyReLU()(outputs)
   outputs = MaxPool2D(pool_size)(outputs)

   return outputs


def PartialLayer(inputs,filters,kernel_size=(3,3)):

   outputs = Conv2D(filters,kernel_size)(inputs)
   outputs = BatchNorm()(outputs)
   outputs = LeakyReLU()(outputs)

   return outputs



def BatchNorm(axis=-1,
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer='zeros',
      gamma_initializer='ones',
      moving_mean_initializer='zeros',
      moving_variance_initializer='ones',
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=True,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      name='BatchNorm_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return layers.BatchNormalization(axis=axis,
      momentum=momentum,
      epsilon=epsilon,
      center=center,
      scale=scale,
      beta_initializer=beta_initializer,
      gamma_initializer=gamma_initializer,
      moving_mean_initializer=moving_mean_initializer,
      moving_variance_initializer=moving_variance_initializer,
      beta_regularizer=beta_regularizer,
      gamma_regularizer=gamma_regularizer,
      beta_constraint=beta_constraint,
      gamma_constraint=gamma_constraint,
      renorm=renorm,
      renorm_clipping=renorm_clipping,
      renorm_momentum=renorm_momentum,
      fused=fused,
      trainable=trainable,
      virtual_batch_size=virtual_batch_size,
      adjustment=adjustment,
      name=name.format(layer_counter))


def Conv2D(filters,
         kernel_size=(3,3),
         strides=(1, 1),
         padding='same',
         data_format='channels_first',
         dilation_rate=(1, 1),
         activation=None,
         use_bias=True,
         kernel_initializer='glorot_uniform',
         bias_initializer='zeros',
         kernel_regularizer=None,
         bias_regularizer=None,
         activity_regularizer=None,
         kernel_constraint=None,
         bias_constraint=None,
         name='Conv2D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return layers.Conv2D(
         filters=filters,
         kernel_size=kernel_size,
         strides=strides,
         padding=padding,
         data_format=data_format,
         dilation_rate=dilation_rate,
         activation=activation,
         use_bias=use_bias,
         kernel_initializer=kernel_initializer,
         bias_initializer=bias_initializer,
         kernel_regularizer=kernel_regularizer,
         bias_regularizer=bias_regularizer,
         activity_regularizer=activity_regularizer,
         kernel_constraint=kernel_constraint,
         bias_constraint=bias_constraint,
         name=name.format(layer_counter))


def MaxPool2D(pool_size,
         strides=None,
         padding='same',
         data_format='channels_first',
         name='MaxPool2D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   if strides is None:
      strides = pool_size
   return layers.MaxPool2D(
         pool_size=pool_size,
         strides=strides,
         padding=padding,
         data_format=data_format,
         name=name.format(layer_counter))


def LeakyReLU(alpha=0.1,name='LeakyRelu_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return layers.LeakyReLU(alpha=alpha,name=name.format(layer_counter))
