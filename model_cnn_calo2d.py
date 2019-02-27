import tensorflow as tf
from tensorflow.python.ops import init_ops
import logging
logger = logging.getLogger(__name__)


layer_counter = 0


def build_model(input_image):

   logger.info('layer %s: %s',layer_counter,input_image)
   output = input_image

   output = Conv3D(32)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,1,2))(output)
   add_summary(output)

   output = Conv3D(32)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,1,2))(output)
   add_summary(output)


   output = Conv3D(64)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,1,2))(output)
   add_summary(output)

   output = Conv3D(64)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,1,2))(output)
   add_summary(output)

   output = Conv3D(128)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,2,2))(output)
   add_summary(output)

   output = Conv3D(128)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,2,2))(output)
   add_summary(output)

   output = Conv3D(256)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((2,2,2))(output)
   add_summary(output)

   output = Conv3D(256)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = MaxPooling3D((1,2,2))(output)
   add_summary(output)

   output = Flatten()(output)
   add_summary(output)


   output = Dense(128)(output)
   add_summary(output)
   output = LeakyRelu(output)
   add_summary(output)

   output = Dense(1)(output)
   add_summary(output)

   output = Sigmoid(output)
   add_summary(output)

   return output


def add_summary(output):
   tf.summary.histogram(output.name.replace(':','_'),output)
   logger.info('layer %s: %s',layer_counter,output)


def Conv3D(filters,
         kernel_size=(3,3,3),
         strides=(1, 1, 1),
         padding='same',
         data_format='channels_last',
         dilation_rate=(1, 1, 1),
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
         name='Conv3D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   return tf.layers.Conv3D(
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


def MaxPooling3D(pool_size,
         strides=None,
         padding='same',
         data_format='channels_last',
         name='MaxPooling3D_{0:02d}'):
   global layer_counter
   layer_counter += 1
   if strides is None:
      strides = pool_size
   return tf.layers.MaxPooling3D(
         pool_size=pool_size,
         strides=strides,
         padding=padding,
         data_format=data_format,
         name=name.format(layer_counter))


def LeakyRelu(inputs):
   global layer_counter
   layer_counter += 1
   return tf.nn.leaky_relu(inputs,name='LeakyRelu_{0:02d}'.format(layer_counter))


def Sigmoid(inputs):
   global layer_counter
   layer_counter += 1
   return tf.nn.sigmoid(inputs,name='Sigmoid_{0:02d}'.format(layer_counter))


def Flatten():
   global layer_counter
   layer_counter += 1
   return tf.layers.Flatten(name='LeakyRelu_{0:02d}'.format(layer_counter))


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

