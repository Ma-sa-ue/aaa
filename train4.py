import ipdb
import os
import pandas as pd
import numpy as np
####import lmdb
from glob import glob
from model4 import *
from util import *

MSCOCO_TRAIN_PATH = '/share/MSCOCO/32x32/train2014/'
SAVE_DURATION = 500

BATCH_SIZE = 128
n_epochs = 200
learning_rate = 0.00005  # 0.00002
image_shape = [32, 32, 3]
dim_z = 100
dim_W1 = 512  # 1024
dim_W2 = 256  # 512
# dim_W3 = 128#256
dim_W4 = 64  # 128
dim_W5 = 3
####dim_channel = 3

visualize_dim = 196

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

lsun_images = glob(MSCOCO_TRAIN_PATH + '*.jpg')

dcgan_model = DCGAN(
    batch_size=BATCH_SIZE,
    image_shape=image_shape,
    dim_z=dim_z,
    dim_W1=dim_W1,
    dim_W2=dim_W2,
    # dim_W3=dim_W3,
    ##dim_channel = 3
    dim_W4=dim_W4,
    dim_W5=dim_W5
)

Z_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen, h_real, h_gen = dcgan_model.build_model()
sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=10)

discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())

train_op_discrim = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

Z_tf_sample, image_tf_sample = dcgan_model.samples_generator(batch_size=visualize_dim)

tf.initialize_all_variables().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dim, dim_z))
iterations = 0
k = 2

p_gen_list = []
p_val_list = []
loss_list = []

for epoch in range(n_epochs):

    for start, end in zip(
            range(0, len(lsun_images), BATCH_SIZE + 15),
            range(BATCH_SIZE + 15, len(lsun_images), BATCH_SIZE + 15)
    ):

        batch_image_files = lsun_images[start:end]
        #batch_images = map(lambda x: crop_resize( os.path.join( MSCOCO_TRAIN_PATH, x) ), batch_image_files)
        # print batch_images.shape[0]
        batch_images = []
        for i in batch_image_files:
            if len(batch_images) == 128:
                break
            ddd = crop_resize(os.path.join(MSCOCO_TRAIN_PATH, i))
            if ddd.shape == (32, 32, 3):
                batch_images.append(ddd)
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, size=[BATCH_SIZE, dim_z]).astype(np.float32)

        p_real_val, p_gen_val, h_real_val, h_gen_val = sess.run([p_real, p_gen, h_real, h_gen], feed_dict={Z_tf: batch_z, image_tf: batch_images})

        if np.mod(iterations, k) != 0:
            _, gen_loss_val = sess.run(
                [train_op_gen, g_cost_tf],
                feed_dict={
                    Z_tf: batch_z,
                })
            print "=========== updating G =========="
            print "iteration:", iterations
            print "gen loss:", gen_loss_val
            loss_list.append(gen_loss_val)
        else:
            _, discrim_loss_val = sess.run(
                [train_op_discrim, d_cost_tf],
                feed_dict={
                    Z_tf: batch_z,
                    image_tf: batch_images
                })
            print "=========== updating D =========="
            print "iteration:", iterations
            print "discrim loss:", discrim_loss_val

            print "real h:", p_real_val.mean(), "  gen h:", p_gen_val.mean()
        p_gen_list.append(p_real_val.mean())
        p_val_list.append(p_gen_val.mean())

        if np.mod(iterations, SAVE_DURATION) == 0:
            generated_samples = sess.run(
                image_tf_sample,
                feed_dict={
                    Z_tf_sample: Z_np_sample
                })
            #####generated_samples = (generated_samples + 1.)/2.
            save_visualization(generated_samples, (14, 14), save_path='./vis10_4/sample_' + str(iterations / SAVE_DURATION) + '.jpg')
        if np.mod(iterations, SAVE_DURATION) == 0:
            np.save("./vis10_4/array1_" + str(iterations / SAVE_DURATION), np.array(p_gen_list))
            np.save("./vis10_4/array2_" + str(iterations / SAVE_DURATION), np.array(p_val_list))
            np.save("./vis10_4/array3_" + str(iterations / SAVE_DURATION), np.array(loss_list))
        iterations += 1
