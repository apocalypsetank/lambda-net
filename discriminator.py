import tensorflow as tf

import tensorflow as tf
import tensorflow.contrib.slim as slim

def attention(x, ch, scope='attention', reuse=False,bs=10):
    with tf.variable_scope(scope, reuse=reuse):
        f = slim.conv2d(x, ch // 8, 1, stride=1, scope='f_conv')
        g = slim.conv2d(x, ch // 8, 1, stride=1, scope='g_conv')
        h = slim.conv2d(x, ch, 1, stride=1, scope='h_conv')

        # N = h * w
        s = tf.matmul(tf.reshape(f, shape=[bs, -1, ch // 8]), tf.reshape(g, shape=[bs, -1, ch // 8]),
                      transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta, tf.reshape(h, shape=[bs, -1, ch]))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape)  # [bs, h, w, C]
        x = gamma * o + x

    return x


def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):
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
        return discriminator(images, is_training=phase_train,
                                   dropout_keep_prob=keep_probability,reuse=reuse)


def discriminator(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        reuse=None,
                        scope='discriminator'):
    end_points = {}

    with tf.variable_scope(scope, 'discriminator', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=2, padding='SAME'):

                net=slim.conv2d(inputs,64,3,stride=2,activation_fn=tf.nn.relu,scope='dis_1')
                net=slim.conv2d(net,128,3,stride=2,activation_fn=tf.nn.relu,scope='dis_2')
		#net = attention(net, 128, scope='att1')
                net=slim.conv2d(net,256,3,stride=2,activation_fn=tf.nn.relu,scope='dis_3')
                net=slim.conv2d(net,512,3,stride=2,activation_fn=tf.nn.relu,scope='dis_4')
                net=slim.flatten(net)
                net=slim.fully_connected(net,1,activation_fn=tf.nn.sigmoid)
    return net



