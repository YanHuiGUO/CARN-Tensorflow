import tensorflow as tf
import tflearn, math
import numpy as np
import tensorflow.contrib.slim as slim
from functools import partial
import tensorflow.contrib.eager as tfe

def group_conv2d(input, out_channels=64, kernel=3, group=1):
    '''
    # input:  A 4-D tensor of shape [batch, height, width, group * in_channel]
    # output: A 4-D tensor of shape [batch, height, width, group * output_channel_per_group]
    '''
    output_channel_per_group = out_channels // group
    shape_dynamic = tf.shape(input)
    shape_static = input.get_shape().as_list()
    group_input = tf.reshape(input,
                                [shape_static[0], shape_static[1], shape_static[2], group, shape_static[3] // group])
    group_output = [slim.conv2d(group_input[:, :, :, i, :], output_channel_per_group, [kernel, kernel], 1, 'SAME',
                                data_format='NHWC', activation_fn=None,
                                weights_initializer= tf.variance_scaling_initializer()) for i in range(group)]
    
    output = tf.concat(group_output, axis=-1)
    return output

def init_weights(modules):
    pass

class MeanShift:
    ## For Image mean normalization
    def __init__(self, mean_rgb, sub):
        sign = -1 if sub else 1
        self.r = mean_rgb[0] * sign
        self.g = mean_rgb[1] * sign
        self.b = mean_rgb[2] * sign

        weight = tf.Variable(tf.reshape(tf.eye(3),
                [1, 1, 3, 3]))
        self.bias = tf.Variable(tf.constant([self.r,self.g,self.b],))

        self.shifter = partial(tf.nn.conv2d,
                        filter=weight,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                        )


    def get_meanshift(self, x):
        x = self.shifter(input = x)
        x = tf.nn.bias_add(x,self.bias)
        return x
        

class BasicBlock:
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, strides=[1, 1, 1, 1], pad='SAME'):

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ksize = ksize
        self.strides = strides
        self.pad = pad
        
    def get_model(self, x):
        weight = tf.Variable(tf.truncated_normal(
                [self.ksize, self.ksize, self.in_channels,self.out_channels],stddev=0.01))
        bias = tf.Variable(tf.zeros([self.out_channels]))
        out = tf.nn.conv2d(
                        input=x,
                        filter=weight,
                        strides=self.strides,
                        padding=self.pad
                        )
        out = tf.nn.bias_add(out, bias)
        out = tf.nn.relu(out)
        return out

class ResidualBlock:
    def __init__(self, 
                in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def get_model(self, x):
        
        weight1 = tf.Variable(tf.truncated_normal(
                [3, 3, self.in_channels, self.out_channels],stddev=0.01))

        bias1 = tf.Variable(tf.zeros([self.out_channels]))
        
        _x = tf.nn.conv2d(
                        input=x,
                        filter=weight1,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                        )
        _x = tf.nn.bias_add(_x, bias1)
        _x = tf.nn.relu(_x)

        weight2 = tf.Variable(tf.truncated_normal(
        [3, 3, self.out_channels, self.out_channels],stddev=0.01))

        bias2 = tf.Variable(tf.zeros([self.out_channels]))
        _x = tf.nn.conv2d(
                        input=_x,
                        filter=weight2,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                        )
        _x = tf.nn.bias_add(_x, bias2)
        out = tf.nn.relu(_x + x)
        return out


class EResidualBlock:
    def __init__(self,in_channels, out_channels, group=1):
        # input:  A 4-D tensor of shape [batch, height, width, group * in_channel]
        # output: A 4-D tensor of shape [batch, height, width, out_channels]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group
        
    def get_model(self, x):
        bias = tf.Variable(tf.zeros([self.out_channels]))
        _x = group_conv2d(
                        input=x,
                        out_channels = self.out_channels,
                        kernel=3,
                        group = self.group
                        )
        _x = tf.nn.bias_add(_x, bias)
        _x = tf.nn.relu(_x)
        _x = group_conv2d(
                        input=_x,
                        out_channels = self.out_channels,
                        kernel=3,
                        group = self.group
                        )
        _x = tf.nn.bias_add(_x, bias)
        _x = tf.nn.relu(_x)

        weight = tf.Variable(tf.truncated_normal([1, 1, self.out_channels, self.out_channels],stddev=0.01))
        bias = tf.Variable(tf.zeros([self.out_channels]))
        _x = tf.nn.conv2d(
                        input=_x,
                        filter=weight,
                        strides=[1, 1, 1, 1],
                        padding='SAME'
                        )
        _x = tf.nn.bias_add(_x, bias)
        out = tf.nn.relu(_x + x)
        return out
        

class _UpsampleBlock:
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        self.n_channels = n_channels
        self.scale = scale
        self.group = group
        
    def get_model(self, x):
        out = x
        if self.scale == 2 or self.scale == 4 or self.scale == 8:
            for _ in range(int(math.log(self.scale, 2))):
                _x = group_conv2d(
                            input=out,
                            out_channels = 4*self.n_channels,
                            kernel=3,
                            group = self.group
                            )
                bias = tf.Variable(tf.zeros([4*self.n_channels]))
                _x = tf.nn.bias_add(_x, bias)
                _x = tf.nn.relu(_x)
                out = tf.depth_to_space(_x,2)

        elif self.scale == 3:
            _x = group_conv2d(
                            input=out,
                            out_channels = 9*self.n_channels,
                            kernel=3,
                            group = self.group
                            )
            bias = tf.Variable(tf.zeros([9*self.n_channels]))
            _x = tf.nn.bias_add(_x, bias)
            _x = tf.nn.relu(_x)
            out = tf.depth_to_space(_x,3)
            
        return out

class UpsampleBlock:
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):

        self.n_channels = n_channels
        self.scale = scale
        self.multi_scale = multi_scale
        self.group = group
        self.up2 = _UpsampleBlock(self.n_channels , scale=2, group=self.group)
        self.up3 = _UpsampleBlock(self.n_channels , scale=3, group=self.group)
        self.up4 = _UpsampleBlock(self.n_channels , scale=4, group=self.group)

        self.up =  _UpsampleBlock(self.n_channels , scale=self.scale, group=self.group)

    def get_model(self, x, scale=1):
        if self.multi_scale:
            if scale == 2:
                return self.up2.get_model(x)
            elif scale == 3:
                return self.up3.get_model(x)
            elif scale == 4:
                return self.up4.get_model(x)
        else:
            return self.up.get_model(x)


   
