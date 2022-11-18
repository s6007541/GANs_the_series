import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import tensorflow_docs.vis.embed as embed

 
epochs = 50
batch_size = 128
lr = 2e-4
b1 = 0.5
b2 = 0.999
latent_dim = 100
img_size = 28
channels = 1
cwd = os.path.abspath(os.getcwd())
savedir = os.path.join(cwd, 'vanilla_epochs')

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
# Normalize images to [-1, 1]
x_train = (x_train - 127.5) / 127.5
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(batch_size)


testnoise = tf.random.normal([20, latent_dim])



# simple Deep NN
def create_generator():
     
    inputs = keras.Input(shape=(100), name='input_layer', batch_size = batch_size)
    # Block 1:input is latent(100), going into a convolution
    
    n_nodes = 128 * 7 * 7
    x = layers.Dense(n_nodes)(inputs)
    x = layers.ReLU(name='relu_0')(x)
    x = layers.Reshape((7, 7, 128))(x)

    
    # Block 2: input is 7 x 7 x (64 * 2)
    x = layers.Conv2DTranspose(64 * 1, kernel_size=3, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(x)
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
     
    # Block 2: input is 14 x 14 x (64 * 1)

    outputs = layers.Conv2DTranspose(1, kernel_size=3, strides= 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_5')(x)
    # Block 3: input is 28 x 28 x (1)
    model = tf.keras.Model(inputs, outputs, name="Generator")
    return model

# simple Deep NN for binary classification task
def create_discriminator():
     
    inputs = keras.Input(shape=(28, 28, 1), name='input_layer', batch_size = batch_size)
    # Block 1: input is 28 x 28 x (1)
    x = layers.Conv2D(64, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_1')(inputs)
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
    model = tf.keras.Model(inputs, outputs, name="Discriminator")
    return model

generator = create_generator()
discriminator = create_discriminator()

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

generator_optimizer = tf.keras.optimizers.Adam(lr = lr, beta_1 = b1, beta_2 = b2 )
discriminator_optimizer = tf.keras.optimizers.Adam(lr = lr, beta_1 = b1, beta_2 = b2 )

@tf.function
def train_step(images,):
    noise = tf.random.normal([batch_size, latent_dim])
 
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
 
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    ## gradients calculation : from 2 different and independent losses
    gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables) 
    gradients_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
     
    ## update params of each model
    # when update params of generator, params of discriminator are frozen.
    # when update params of discriminator, params of generator are frozen.
    generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_disc,discriminator.trainable_variables))
    return gen_loss,disc_loss
    

def train(dataset, epochs, savefig = False):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            gen_loss,disc_loss = train_step(image_batch)

        print ('epoch {} takes {:.7f} sec. G Loss {:.7f}. D Loss {:.7f}.'.format(epoch + 1, time.time()-start, gen_loss,disc_loss))
        if savefig == True:
            # %matplotlib inline
            generated_test_images = generator([testnoise])
#             print(generated_test_images.shape)
            fig, axes = plt.subplots(2,10, figsize = (20,4))
            for i,ax in enumerate(axes.flat):
                ax.imshow(generated_test_images[i],cmap='gray')
                ax.axis("off")
            fig.suptitle("epoch {}".format(epoch+1), fontsize = 20)
            
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            plt.savefig(os.path.join(savedir, 'dc_epoch_{:04d}.png'.format(epoch+1)))
            plt.close(fig)
            
train(train_dataset, epochs, True)

# testnoise = tf.random.normal([20, latent_dim])
generated_test_images = generator(testnoise)


## visualize generated data
fig, axes = plt.subplots(2,10, figsize = (20,4))
for i,ax in enumerate(axes.flat):
    ax.imshow(generated_test_images[i],cmap='gray')
    ax.set_title("final epoch",{'fontsize': 10})
    ax.axis("off")
    
    
anim_file = 'dcgan.gif'
import glob
import imageio.v2
with imageio.get_writer(anim_file, mode='I') as writer:
    
    filenames = glob.glob(os.path.join(savedir,'./dc_epoch_*.png' ))
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
embed.embed_file(anim_file)
