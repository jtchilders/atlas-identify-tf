import tensorflow as tf
from tensorflow.python.ops import init_ops
import logging,sys
logger = logging.getLogger(__name__)


layer_counter = 0


def build_model(input_image,num_boxes,num_labels,batch_size):
   logger.info('building model, input image: %s',input_image)
   outputs = input_image

   # Layer 1
   outputs = FullLayer(outputs,32,(3,3),(2,2))

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
   outputs = PartialLayer(outputs,512,(3,3))

   connection = outputs

   outputs = MaxPooling2D((2,2))(outputs)

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
   connection = Conv2D(64,(1,1))(connection)
   add_summary(connection)
   connection = BatchNorm()(connection)
   connection = LeakyReLU(connection)
   total_pixels = connection.get_shape()[1].value*connection.get_shape()[2].value*connection.get_shape()[3].value

   new_axis2 = outputs.get_shape()[2].value
   new_axis3 = outputs.get_shape()[3].value
   new_axis1 = int(total_pixels/new_axis2/new_axis3)
   connection = tf.reshape(connection,
         shape=(batch_size,new_axis1,new_axis2,new_axis3),
         name='Reshape_%s'%layer_counter)
   # connection = tf.space_to_depth(connection,block_size=2,data_format='NCHW')

   logger.info('outputs = %s; connection = %s',outputs,connection)

   outputs = tf.concat([connection,outputs],axis=1)

   logger.info('concat = %s',outputs)
   
   # Layer 22
   outputs = Conv2D(1024, (3,3))(outputs)
   outputs = BatchNorm()(outputs)
   outputs = LeakyReLU(outputs)

   grid_h = outputs.get_shape()[2].value
   grid_w = outputs.get_shape()[3].value

   logger.info('grid_h x grid_w =  %s x %s ',grid_h,grid_w)

   # make the object detection layer
   outputs = Conv2D(num_boxes * (4 + 1 + num_labels),
                  (1,1), strides=(1,1),
                  padding='same',
                  name='DetectionLayer_%s' % layer_counter,
                  kernel_initializer='lecun_normal',
                  data_format='channels_first')(outputs)
   add_summary(outputs)
   outputs = tf.reshape(outputs,shape=(batch_size,grid_h, grid_w, num_boxes, 4 + 1 + num_labels),name='Reshape_%s'%layer_counter)

   add_summary(outputs)

   return outputs,grid_h,grid_w


def add_summary(output):
   tf.summary.histogram(output.name.replace(':','_'),output)
   logger.info('layer %s: %s',layer_counter,output)
   sys.stdout.flush()
   sys.stderr.flush()


def FullLayer(inputs,filters,kernel_size=(3,3),pool_size=(2,2)):

   outputs = Conv2D(filters,kernel_size)(inputs)
   add_summary(outputs)
   outputs = BatchNorm()(outputs)
   outputs = LeakyReLU(outputs)
   outputs = MaxPooling2D(pool_size)(outputs)
   add_summary(outputs)

   return outputs


def PartialLayer(inputs,filters,kernel_size=(3,3)):

   outputs = Conv2D(filters,kernel_size)(inputs)
   add_summary(outputs)
   outputs = BatchNorm()(outputs)
   outputs = LeakyReLU(outputs)
   add_summary(outputs)

   return outputs



def BatchNorm(axis=-1,
      momentum=0.99,
      epsilon=0.001,
      center=True,
      scale=True,
      beta_initializer=tf.zeros_initializer(),
      gamma_initializer=tf.ones_initializer(),
      moving_mean_initializer=tf.zeros_initializer(),
      moving_variance_initializer=tf.ones_initializer(),
      beta_regularizer=None,
      gamma_regularizer=None,
      beta_constraint=None,
      gamma_constraint=None,
      renorm=False,
      renorm_clipping=None,
      renorm_momentum=0.99,
      fused=None,
      trainable=True,
      virtual_batch_size=None,
      adjustment=None,
      name='BatchNorm_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return tf.layers.BatchNormalization(axis=axis,
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
         use_bias=False,
         kernel_initializer=None,
         bias_initializer=init_ops.zeros_initializer(),
         kernel_regularizer=None,
         bias_regularizer=None,
         activity_regularizer=None,
         kernel_constraint=None,
         bias_constraint=None,
         trainable=True,
         name='Conv2D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return tf.layers.Conv2D(
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
         trainable=trainable,
         name=name.format(layer_counter))


def MaxPooling2D(pool_size,
         strides=None,
         padding='same',
         data_format='channels_first',
         name='MaxPool2D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   if strides is None:
      strides = pool_size
   return tf.layers.MaxPooling2D(
         pool_size=pool_size,
         strides=strides,
         padding=padding,
         data_format=data_format,
         name=name.format(layer_counter))


def LeakyReLU(inputs,alpha=0.1,name='LeakyRelu_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return tf.nn.leaky_relu(inputs,alpha=alpha,name=name.format(layer_counter))


def Sigmoid(inputs):
   global layer_counter
   layer_counter += 1
   return tf.nn.sigmoid(inputs,name='Sigmoid_{0:02d}'.format(layer_counter))


def Flatten():
   global layer_counter
   layer_counter += 1
   return tf.layers.Flatten(name='Flatten_{0:02d}'.format(layer_counter))


def Dense(units,
         activation=None,
         use_bias=True,
         kernel_initializer=None,
         bias_initializer=init_ops.zeros_initializer(),
         kernel_regularizer=None,
         bias_regularizer=None,
         activity_regularizer=None,
         kernel_constraint=None,
         bias_constraint=None,
         trainable=True,
         name='Dense_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return tf.layers.Dense(units=units,
         activation=activation,
         use_bias=use_bias,
         kernel_initializer=kernel_initializer,
         bias_initializer=bias_initializer,
         kernel_regularizer=kernel_regularizer,
         bias_regularizer=bias_regularizer,
         activity_regularizer=activity_regularizer,
         kernel_constraint=kernel_constraint,
         bias_constraint=bias_constraint,
         trainable=trainable,
         name=name.format(layer_counter))

