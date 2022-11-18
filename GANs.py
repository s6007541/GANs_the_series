import os
import time
import tensorflow as tf
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
def create_generator(image_dim):
     
    inputs = layers.Input(shape=(100,))
    x = layers.Dense(128, kernel_initializer=tf.keras.initializers.he_uniform)(inputs)
    print(x.dtype)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256, kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(512, kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(1024, kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8)(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(image_dim, activation='tanh', kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    outputs = tf.reshape(x, [-1, img_size, img_size, channels], name=None)
    model = tf.keras.Model(inputs, outputs, name="Generator")
    return model

# simple Deep NN for binary classification task
def create_discriminator():
    inputs = layers.Input(shape=(img_size, img_size, channels))
    reshape = tf.reshape(inputs, [-1, 784], name=None)
    x = layers.Dense(512, kernel_initializer=tf.keras.initializers.he_uniform)(reshape)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dense(256, kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    x = layers.LeakyReLU(0.2)(x)
    outputs = layers.Dense(1, activation='sigmoid', kernel_initializer=tf.keras.initializers.he_uniform)(x) 
    model = tf.keras.Model(inputs, outputs, name="Discriminator")
    return model

generator = create_generator(28*28*1)
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
            plt.savefig(os.path.join(savedir, 'vanilla_epoch_{:04d}.png'.format(epoch+1)))
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
    
    
anim_file = 'gan.gif'
import glob
import imageio.v2
with imageio.get_writer(anim_file, mode='I') as writer:
    
    filenames = glob.glob(os.path.join(savedir,'./vanilla_epoch_*.png' ))
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)
        image = imageio.v2.imread(filename)
        writer.append_data(image)
embed.embed_file(anim_file)
