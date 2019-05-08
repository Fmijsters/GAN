from IPython import display

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from utils import Logger

import tensorflow as tf

import numpy as np

DATA_FOLDER = './tf_data/VGAN/MNIST'
IMAGE_PIXELS = 28*28
NOISE_SIZE = 100
BATCH_SIZE = 100

def noise(n_rows, n_cols):
    return np.random.normal(size=(n_rows, n_cols))

def xavier_init(size):
    in_dim = size[0] if len(size) == 1 else size[1]
    stddev = 1. / np.sqrt(float(in_dim))
    return tf.random_uniform(shape=size, minval=-stddev, maxval=stddev)

def images_to_vectors(images):
    
    return images.reshape(images.shape[0], 784)

def vectors_to_images(vectors):
    return vectors.reshape(vectors.shape[0], 28, 28, 1)


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ])
    out_dir = '{}/dataset'.format(DATA_FOLDER)
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

## Discriminator
class DiscriminatorNet():
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        # Input
        self.X = tf.placeholder(tf.float32, shape=(None, IMAGE_PIXELS))

        # Layer 1 Variables
        self.D_W1 = tf.Variable(xavier_init([784, 1024]))
        self.D_B1 = tf.Variable(xavier_init([1024]))

        # Layer 2 Variables
        self.D_W2 = tf.Variable(xavier_init([1024, 512]))
        self.D_B2 = tf.Variable(xavier_init([512]))

        # Layer 3 Variables
        self.D_W3 = tf.Variable(xavier_init([512, 256]))
        self.D_B3 = tf.Variable(xavier_init([256]))

        # Out Layer Variables
        self.D_W4 = tf.Variable(xavier_init([256, 1]))
        self.D_B4 = tf.Variable(xavier_init([1]))

        # Store Variables in list
        self.var_list = [self.D_W1, self.D_B1, self.D_W2, self.D_B2, self.D_W3, self.D_B3, self.D_W4, self.D_B4]

    def forward(self,x):
        l1 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(x,   self.D_W1) + self.D_B1, .2), .3)
        l2 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l1,  self.D_W2) + self.D_B2, .2), .3)
        l3 = tf.nn.dropout(tf.nn.leaky_relu(tf.matmul(l2,  self.D_W3) + self.D_B3, .2), .3)
        out = tf.matmul(l3, self.D_W4) + self.D_B4
        return out

## Generator
class GeneratorNet():
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        self.Z = tf.placeholder(tf.float32, shape=(None, NOISE_SIZE))

        # Layer 1 Variables
        self.G_W1 = tf.Variable(xavier_init([100, 256]))
        self.G_B1 = tf.Variable(xavier_init([256]))

        # Layer 2 Variables
        self.G_W2 = tf.Variable(xavier_init([256, 512]))
        self.G_B2 = tf.Variable(xavier_init([512]))

        # Layer 3 Variables
        self.G_W3 = tf.Variable(xavier_init([512, 1024]))
        self.G_B3 = tf.Variable(xavier_init([1024]))

        # Out Layer Variables
        self.G_W4 = tf.Variable(xavier_init([1024, 784]))
        self.G_B4 = tf.Variable(xavier_init([784]))

        # Store Variables in list
        self.var_list = [self.G_W1, self.G_B1, self.G_W2, self.G_B2, self.G_W3, self.G_B3, self.G_W4, self.G_B4]

    def forward(self,z):
        l1 = tf.nn.leaky_relu(tf.matmul(z,  self.G_W1) + self.G_B1, .2)
        l2 = tf.nn.leaky_relu(tf.matmul(l1, self.G_W2) + self.G_B2, .2)
        l3 = tf.nn.leaky_relu(tf.matmul(l2, self.G_W3) + self.G_B3, .2)
        out = tf.nn.tanh(tf.matmul(l3, self.G_W4) + self.G_B4)
        return out
    
generator_net = GeneratorNet()
discriminator_net = DiscriminatorNet()


# Load data
data = mnist_data()
# Create loader with data, so that we can iterate over it
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
# Num batches
num_batches = len(data_loader)

G_sample = generator_net.forward(generator_net.Z)

D_real = discriminator_net.forward(discriminator_net.X)

D_fake = discriminator_net.forward(G_sample)

# Losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))

# Optimizers
D_opt = tf.train.AdamOptimizer(2e-4).minimize(D_loss, var_list=discriminator_net.var_list)
G_opt = tf.train.AdamOptimizer(2e-4).minimize(G_loss, var_list=generator_net.var_list)

num_test_samples = 16
test_noise = noise(num_test_samples, NOISE_SIZE)

num_epochs = 200

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()
# Init Logger
logger = Logger(model_name='DCGAN1', data_name='CIFAR10')

# Iterate through epochs
for epoch in range(num_epochs):
    for n_batch, (batch,_) in enumerate(data_loader):
        
        # 1. Train Discriminator
        X_batch = images_to_vectors(batch.permute(0, 2, 3, 1).numpy())
        feed_dict = {discriminator_net.X: X_batch, generator_net.Z: noise(BATCH_SIZE, NOISE_SIZE)}
        _, d_error, d_pred_real, d_pred_fake = session.run(
            [D_opt, D_loss, D_real, D_fake], feed_dict=feed_dict
        )

        # 2. Train Generator
        feed_dict = {generator_net.Z: noise(BATCH_SIZE, NOISE_SIZE)}
        _, g_error = session.run(
            [G_opt, G_loss], feed_dict=feed_dict
        )

        if n_batch % 100 == 0:
            display.clear_output(True)
            # Generate images from test noise
            test_images = session.run(
                G_sample, feed_dict={generator_net.Z: test_noise}
            )
            test_images = vectors_to_images(test_images)
            # Log Images
            logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches, format='NHWC');
            # Log Status
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )