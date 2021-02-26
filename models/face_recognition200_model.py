import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.keras.layers import *

class CNN_Block(tf.keras.Model):
  def __init__(self, num_filters, conv_size):
    super(CNN_Block, self).__init__()
    self.conv = Conv2D(num_filters, conv_size, padding='same')
    self.bn = BatchNormalization()
    self.pool = MaxPool2D()

  def call(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = tf.nn.relu(x)
    x = self.pool(x)
    return x

class Encoder(tf.keras.Model):
  def __init__(self):
    super(Encoder, self).__init__()
    self.block1 = CNN_Block(32, 3)
    self.block2 = CNN_Block(64, 3)
    self.block3 = CNN_Block(128, 3)
    self.block4 = CNN_Block(256, 3)
    self.flatten = Flatten()
    self.dense = Dense(1024, activation='relu')

  def call(self, x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.flatten(x)
    x = self.dense(x)
    return x

class Euclidean_Distance(tf.keras.Model):
  def __init__(self):
    super(Euclidean_Distance, self).__init__()
    self.dense1 = Dense(512)
    self.dense2 = Dense(1, activation='sigmoid')

  def call(self, xA, xB):
    xA = self.dense1(xA)
    xA = tf.nn.leaky_relu(xA)
    xB = self.dense1(xB)
    xB = tf.nn.leaky_relu(xB)
    dist = xA-xB
    x = self.dense2(dist)
    return x


def build_model():
  imgA = Input((64, 64, 3), dtype=tf.float32, name='inputA')
  imgB = Input((64, 64, 3), dtype=tf.float32, name='inputB')
  encoder = Encoder()
  featA = encoder(imgA)
  featB = encoder(imgB)
  dist = Euclidean_Distance()
  outputs = dist(featA, featB)
  model = tf.keras.Model(inputs=(imgA, imgB), outputs=outputs)
  optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics='accuracy')
  #model.summary()
  return model

model = build_model()
model.load_weights('models/model_files/face_recognition(200).h5')


