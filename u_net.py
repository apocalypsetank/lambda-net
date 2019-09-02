import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.io as sio
import matplotlib.pyplot as plt




def inference(images, keep_probability, phase_train=True,
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        'scale':True,
        'is_training':phase_train,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
    }

    with slim.arg_scope([slim.conv2d, slim.fully_connected,slim.conv2d_transpose],
                        weights_initializer=slim.initializers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return encoder_decoder(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability,reuse=reuse)


def encoder_decoder(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='generator'):
    end_points = {}

    with tf.variable_scope(scope, 'generator', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):


                ##################### encoder ##############################################
                net = slim.conv2d(inputs, 32, 3, stride=1, padding='SAME',scope='en_1_1')
                net=slim.conv2d(net, 32, 3, stride=1, padding='SAME',scope='en_1_2')

                end_points['encode_1'] = net #bs*200*512*32
                net=slim.max_pool2d(net,2,stride=2,padding='SAME',scope='Pool1')
                #bs*100*256*32


                net = slim.conv2d(net, 64, 3, stride=1, padding='SAME', scope='en_2_1')
                net = slim.conv2d(net,64, 3, stride=1, padding='SAME', scope='en_2_2')

                end_points['encode_2'] = net#(bs, 50, 135, 64)
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool2')
                #(bs, 50, 128, 64)

                net = slim.conv2d(net, 128, 3, stride=1, padding='SAME', scope='en_3_1')
                net = slim.conv2d(net,128, 3, stride=1, padding='SAME', scope='en_3_2')
                end_points['encode_3'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='Pool3')
                #(bs, 25, 64, 128)

                #
                net = slim.conv2d(net, 256, 3, stride=1, padding='SAME', scope='en_4_1')
                net = slim.conv2d(net,256, 3, stride=1, padding='SAME', scope='en_4_2')#(bs, 12, 34, 256)
                end_points['encode_4'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool4')
                # (bs, 13, 32, 256)

                net=slim.conv2d(net, 512, 3, stride=1, padding='SAME', scope='en_5_1')
                net=slim.conv2d(net, 512, 3, stride=1, padding='SAME', scope='en_5_2')
                end_points['encode_5'] = net
                net = slim.max_pool2d(net, 2, stride=2, padding='SAME', scope='Pool5')


                net=slim.conv2d(net, 1024, 3, stride=1, padding='SAME', scope='en_6')
                #
                net = slim.conv2d(net, 1024, 3, stride=1, padding='SAME', scope='en_7')
                # ##################### encoder ##############################################
                net = slim.conv2d_transpose(net, 512, 2, 2, padding='VALID')
                net=tf.concat([net,end_points['encode_5']],3)
                net = slim.conv2d(net, 512, 3, stride=1)
                net = slim.conv2d(net, 512, 3, stride=1)



                net=slim.conv2d_transpose(net,256,2,2,padding='VALID')
                net=tf.concat([net,end_points['encode_4']],3)
                net=slim.conv2d(net,256,3,stride=1)
                net=slim.conv2d(net,256,3,stride=1)

                #(bs,25,64,256)

                net=slim.conv2d_transpose(net,128,2,2,padding='VALID')
                net = tf.concat([net, end_points['encode_3']], 3)
                net = slim.conv2d(net, 128, 3, stride=1)
                net = slim.conv2d(net, 128, 3, stride=1)


                #(bs, 50, 128, 128)


                net=slim.conv2d_transpose(net,64,2,2,padding='SAME')
                net = tf.concat([net, end_points['encode_2']], 3)
                net = slim.conv2d(net, 64, 3, stride=1)
                net = slim.conv2d(net, 64, 3, stride=1)

                #bs,100,256,64

                net = slim.conv2d_transpose(net, 32, 2, 2, padding='SAME')
                net = tf.concat([net, end_points['encode_1']], 3)
                net = slim.conv2d(net, 32, 3, stride=1)
                net = slim.conv2d(net, 32, 3, stride=1)

                # bs,200,512,32

                net=slim.conv2d(net,1,1,stride=1,activation_fn=None)
                net=net+inputs
    return net
