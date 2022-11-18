#import the required packages
import os
import time
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import glob
import imageio.v2
import tensorflow_docs.vis.embed as embed


epochs = 50
batch_size = 128
lr = 2e-4
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 28
channels = 1
n_classes = 10
cwd = os.path.abspath(os.getcwd())
savedir = os.path.join(cwd, 'cond_epochs')


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, y_train = shuffle(x_train, y_train.reshape(-1,1))
# print(x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_train = (x_train - 127.5) / 127.5 # Normalize the images to [-1, 1]
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size)
y_dataset = tf.data.Dataset.from_tensor_slices(y_train).batch(batch_size)


classtarget = tf.reshape(tf.constant([[0,0,1.0,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]]), (-1,1))
testnoise = tf.random.normal([20, latent_dim])


# label input
con_label = layers.Input(shape=(1,))
 
# latent vector input
latent_vector = layers.Input(shape=(100,))

def label_conditioned_generator(n_classes=n_classes, embedding_dim=100):
    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label)
    #print(label_embedding)
    # linear multiplication
    nodes = 7 * 7
    label_dense = layers.Dense(nodes)(label_embedding)
    # reshape to additional channel
    label_reshape_layer = layers.Reshape((7, 7, 1))(label_dense)
    return label_reshape_layer
 
def latent_input(latent_dim=100):
    # image generator input
    nodes = 128 * 7 * 7
    latent_dense = layers.Dense(nodes)(latent_vector)
    latent_dense = layers.ReLU()(latent_dense)
    latent_reshape = layers.Reshape((7, 7, 128))(latent_dense)
    return latent_reshape

# define the final generator model
def define_generator():
    latent_vector_output = label_conditioned_generator()
    label_output = latent_input()
    # merge label_conditioned_generator and latent_input output #######
    merge = layers.Concatenate()([latent_vector_output, label_output])
     
    # Block 2: input is 7 x 7 x (64 * 2)
    x = layers.Conv2DTranspose(64 * 1, kernel_size=3, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
     
    # Block 2: input is 14 x 14 x (64 * 1)
        
    out_layer = layers.Conv2DTranspose(1, kernel_size=3, strides= 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_5')(x)
    # Block 3: input is 28 x 28 x (1)
#     model = tf.keras.Model(inputs, outputs, name="Generator")
    
    
    
   # define model
    model = tf.keras.Model([latent_vector, con_label], out_layer)
    return model

# label input
con_label_disc = layers.Input(shape=(1,))
 
# input image
inp_img_disc = layers.Input(shape=(28,28,1))

def label_condition_disc(in_shape=(28,28,1), n_classes=n_classes, embedding_dim=100):

    # embedding for categorical input
    label_embedding = layers.Embedding(n_classes, embedding_dim)(con_label_disc)
    nodes = in_shape[0] * in_shape[1] * in_shape[2]
    # scale up to image dimensions with linear layer
    label_dense = layers.Dense(nodes)(label_embedding)
    # reshape to a tensor
    label_reshape_layer = layers.Reshape((in_shape[0], in_shape[1], 1))(label_dense)
    # image input
    return label_reshape_layer



def image_disc(in_shape=(28,28,1)):
    return inp_img_disc

def define_discriminator():
    global con_label_disc,inp_img_disc
    label_condition_output = label_condition_disc()
    inp_image_output = image_disc()
    # concat label as a channel
    merge = layers.Concatenate()([inp_image_output, label_condition_output])
     
    x = layers.Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(merge)
    x = layers.LeakyReLU(0.2, name='leaky_relu_1')(x)
     
    # Block 2: input is 14 x 14 x (64)
    x = layers.Conv2D(64 * 2, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_2')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_2')(x)
     
    # Block 3: input is 7 x 7 x (64*2)
    x = layers.Conv2D(64 * 4, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_3')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_3')(x)
   
    # Block 4: input is 4 x 4 x (64*4)
    x = layers.Conv2D(64 * 8, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_4')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.LeakyReLU(0.2, name='leaky_relu_4')(x)
     
    # Block 5: input is 2 x 2 x (64*4)
    outputs = layers.Conv2D(1, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='sigmoid', name='conv_5')(x)
    # Output: 1 x 1 x 1
   
  
#     flattened_out = layers.Flatten()(x)
#     # dropout
#     dropout = layers.Dropout(0.4)(flattened_out)
#     # output
#     dense_out = layers.Dense(1, activation='sigmoid')(dropout)
    # define model
 
    # define model
    model = tf.keras.Model([inp_img_disc, con_label_disc], outputs)
    return model

conditional_gen = define_generator()
conditional_discriminator = define_discriminator()


# reference https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
# bincrossentropy(y_true, y_pred)
bincrossentropy = tf.keras.losses.BinaryCrossentropy()


# train generator to generate more realistic fake output 
def generator_loss(fake_output):
    # next time : fake output will be likely to classify to 1
    gen_loss = bincrossentropy(tf.ones_like(fake_output), fake_output)
    return gen_loss

# train discriminator to know that real output is real, fake output is fake
def discriminator_loss(real_output, fake_output):
    # next time : real output will be likely to classify to 1
    real_loss = bincrossentropy(tf.ones_like(real_output), real_output)
    # next time : fake output will be likely to classify to 0
    fake_loss = bincrossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

generator_optimizer = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )
discriminator_optimizer = tf.keras.optimizers.Adam(lr = 0.0002, beta_1 = 0.5, beta_2 = 0.999 )


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".

@tf.function
def train_step(images,target):
    noise = tf.random.normal([images.shape[0], latent_dim])
 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = conditional_gen([noise,target], training=True)
 
        real_output = conditional_discriminator([images,target], training=True)
        fake_output = conditional_discriminator([generated_images,target], training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    ## gradients calculation : from 2 different and independent losses
    gradients_of_gen = gen_tape.gradient(gen_loss, conditional_gen.trainable_variables) 
    gradients_of_disc = disc_tape.gradient(disc_loss, conditional_discriminator.trainable_variables)
     
    ## update params of each model
    # when update params of generator, params of discriminator are frozen.
    # when update params of discriminator, params of generator are frozen.
    generator_optimizer.apply_gradients(zip(gradients_of_gen, conditional_gen.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc,conditional_discriminator.trainable_variables))
    return gen_loss,disc_loss


def train(dataset, epochs, savefig = False):
    
    for epoch in range(epochs):
        start = time.time()
        i = 1
        for image_batch,y_batch in zip(dataset,y_dataset):
            img = tf.cast(image_batch, tf.float32)
            y = tf.cast(y_batch, tf.float32)
            gen_loss, disc_loss = train_step(img,y)
            i += 1
#             if i == 100:
#                 break
        print ('epoch {} takes {:.7f} sec. G Loss {:.7f}. D Loss {:.7f}. '.format(epoch + 1, time.time()-start, gen_loss,disc_loss))
        if savefig == True:
            # %matplotlib inline
            testnoise = tf.random.normal([20, latent_dim])
            generated_test_images = conditional_gen([testnoise, classtarget])
#             print(generated_test_images.shape)
            fig, axes = plt.subplots(2,10, figsize = (20,4))
            for i,ax in enumerate(axes.flat):
                ax.imshow(generated_test_images[i],cmap='gray')
                ax.axis("off")
            fig.suptitle("epoch {}".format(epoch), fontsize = 20)
            
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            plt.savefig(os.path.join(savedir, 'cond_epoch_{:04d}.png'.format(epoch+1)))
            plt.close(fig)
            

train(train_dataset, epochs, True)

# testnoise = tf.random.normal([20, latent_dim])
generated_test_images = conditional_gen([testnoise,classtarget])


## visualize generated data
fig, axes = plt.subplots(2,10, figsize = (20,4))
for i,ax in enumerate(axes.flat):
    ax.imshow(generated_test_images[i],cmap='gray')
    ax.set_title("final epoch",{'fontsize': 10})
    ax.axis("off")
    
    
anim_file = 'cgan.gif'


with imageio.get_writer(anim_file, mode='I') as writer:
    
    filenames = glob.glob(os.path.join(savedir,'./cond_epoch_*.png' ))
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
embed.embed_file(anim_file)