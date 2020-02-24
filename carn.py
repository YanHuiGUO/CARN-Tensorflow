import tensorflow as tf
import carnops as ops
from functools import partial

class Block:
    def __init__(self, 
                 in_channels, out_channels,
                 group=1):

        self.b1 = ops.ResidualBlock(64, 64)
        self.b2 = ops.ResidualBlock(64, 64)
        self.b3 = ops.ResidualBlock(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, [1, 1, 1, 1], 'SAME')
        self.c2 = ops.BasicBlock(64*3, 64, 1, [1, 1, 1, 1], 'SAME')
        self.c3 = ops.BasicBlock(64*4, 64, 1, [1, 1, 1, 1], 'SAME')

    def get_model(self, x):
        c0 = o0 = x

        b1 = self.b1.get_model(o0)
        c1 = tf.concat([c0, b1], axis=-1)

        o1 = self.c1.get_model(c1)
        
        b2 = self.b2.get_model(o1)
        c2 = tf.concat([c1, b2], axis=-1)
        o2 = self.c2.get_model(c2)
        
        b3 = self.b3.get_model(o2)
        c3 = tf.concat([c2, b3], axis=-1)
        o3 = self.c3.get_model(c3)

        return o3

class Net:
    def __init__(self, **kwargs):
        
        scale = kwargs.get("scale")
        multi_scale = kwargs.get("multi_scale")
        group = kwargs.get("group", 1)

        self.sub_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = ops.MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.entry = partial(tf.layers.conv2d,
                        filters=64,
                        kernel_size=3,
                        strides=(1,1),
                        padding='SAME') #nn.Conv2d(3, 64, 3, 1, 1)
        
        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = ops.BasicBlock(64*2, 64, 1, [1, 1, 1, 1], 'SAME')
        self.c2 = ops.BasicBlock(64*3, 64, 1, [1, 1, 1, 1], 'SAME')
        self.c3 = ops.BasicBlock(64*4, 64, 1, [1, 1, 1, 1], 'SAME')
        
        self.upsample = ops.UpsampleBlock(64, scale=scale, 
                                          multi_scale=multi_scale,
                                          group=group)
        self.exit = partial(tf.layers.conv2d,
                        filters=3,
                        kernel_size=3,
                        strides=(1,1),
                        padding='SAME',
                        name='output'
                        ) #nn.Conv2d(64, 3, 3, 1, 1)
                
    def get_model(self, x, scale):
        # x = self.sub_mean.get_meanshift(x)
        x = self.entry(inputs = x)
        
        c0 = o0 = x
        

        b1 = self.b1.get_model(o0)
        # print (b1)
        c1 = tf.concat([c0, b1], axis=-1)
        o1 = self.c1.get_model(c1)
        
        b2 = self.b2.get_model(o1)
        c2 = tf.concat([c1, b2], axis=-1)
        o2 = self.c2.get_model(c2)
        
        b3 = self.b3.get_model(o2)
        c3 = tf.concat([c2, b3], axis=-1)
        o3 = self.c3.get_model(c3)

        out = self.upsample.get_model(o3, scale=scale)

        out = self.exit(inputs = out)
        # out = self.add_mean.get_meanshift(out)

        return out
if __name__ == '__main__':
    
    inp = tf.ones([1, 64,64,3],  dtype=tf.float32) 
    test_net = Net(scale=4,multi_scale=True,group=1)
    out = test_net.get_model(inp,scale=2)
    print (out)


