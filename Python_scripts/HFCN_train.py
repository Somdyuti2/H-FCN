'''
Copyright (c) 2019 Netflix, Inc., University of Texas at Austin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Attribution: the H-FCN model design was motivated by the B-CNN paper 
(https://arxiv.org/abs/1709.09890) and corresponding code repository 
(https://github.com/zhuxinqimac/B-CNN). 

Authors:
Somdyuti Paul <somdyuti@utexas.edu>
Andrey Norkin <anorkin@netflix.com>
Alan C. Bovik <bovik@ece.utexas.edu>
'''

import os
import numpy as np
import keras.backend as K
import keras
from keras import optimizers
from keras.utils import Sequence
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Activation, Input, Concatenate, Lambda
import glob
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import tensorflow as tf
import argparse

class TrainValTensorBoard(keras.callbacks.TensorBoard):
    def __init__(self, log_dir='./logs_fcn', **kwargs):
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()


def get_image(img_path):
  im =np.fromfile(img_path, count=4356, dtype='uint8')
  im=im.reshape([66, 66,1])
  im=im.astype('float')
  # im = (im.astype('float')-m)/sd
  im=im[2:,2:]
  return np.asarray(im)

def get_qp1(path):
    q=np.load(path)['q']
    qp1 = np.empty([8, 8, 1])
    qp1.fill(q)
    return qp1

def get_partition(path):
  p0=np.load(path)['L0']
  p1=np.load(path)['L1']
  p2=np.load(path)['L2']
  p3 = np.load(path)['L3']

  p0=np.reshape(p0,1)
  p1=np.reshape(p1,4)
  p2=np.reshape(p2,16)
  p3 = np.reshape(p3, 64)
  partition=[]
  partition.append(p0)
  for i in range(len(p1)):
      partition.append(p1[i])

  for i in range(len(p2)):
      partition.append(p2[i])

  for i in range(len(p3)):
      partition.append(p3[i])

  return np.asarray(partition)

class Data_Generator(Sequence):

    def __init__(self, image_filenames, labels, batch_size, shuffle=True):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        x=np.empty([self.batch_size,64,64,1])
        y=np.empty([self.batch_size,85])
        qp1=np.empty([self.batch_size,8,8,1])
        # P=np.empty([self.batch_size,4,85])
        c=0
        for i, j in zip(batch_x, batch_y):
            x[c] = np.array(get_image(i))
            y[c] = np.array(get_partition(j))
            qp1[c]=np.array(get_qp1(j))
            c+=1

        partition = []
        for i in range(np.shape(y)[1]):
            partition.append(to_categorical(y[:, i], 4))
        return [x,qp1], partition

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


def load_data(X_samples, y_samples, batch_size=32):
    while True:
        num_batches = int(len(X_samples) / batch_size)
        X_batches = np.array_split(X_samples, num_batches)
        Y_batches = np.array_split(y_samples, num_batches)
        for b in range(len(X_batches)):

            x = np.array(list(map(get_image, X_batches[b])))
            y = np.array(list(map(get_partition, Y_batches[b])))
            qp1=np.array(list(map(get_qp1, Y_batches[b])))

            partition=[]
            for i in range(np.shape(y)[1]):
                partition.append(to_categorical(y[:,i],4))

            yield [x, qp1], partition

def load_validation(val_path):
    val_names = glob.glob(val_path+"/*.png")
    val_names = np.asarray(val_names)
    np.random.shuffle(val_names)

    val_y = np.asarray([w.replace('png', 'npz') for w in val_names])
    partition = []
    x = np.array(list(map(get_image, val_names)))
    y = np.array(list(map(get_partition, val_y)))
    qp1 = np.array(list(map(get_qp1, val_y)))

    for i in range(np.shape(y)[1]):
        partition.append(to_categorical(y[:, i], 4))
    return x, qp1, partition



def cnn_model():
    img_rows, img_cols = 64, 64
    if K.image_data_format() == 'channels_first':
        input_shape = (1, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 1)

    img_input = Input(shape=input_shape, name='input')
    qp_input1 = Input(shape=(8,8,1), name='qp1')


    qp_input2 = Lambda(lambda x: x[:, 0:4, 0:4, :], name="Lambda_qp2")(qp_input1)
    qp_input3 = Lambda(lambda x: x[:, 0:2, 0:2, :], name="Lambda_qp3")(qp_input1)
    qp_input4 = Lambda(lambda x: x[:, 0:1, 0:1, :], name="Lambda_qp4")(qp_input1)


    # block 1
    x = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3),  kernel_initializer='he_uniform', activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) #output size 32 x32 x16

    # branch 1
    c_1_bch = Conv2D(32, (4, 4), strides=(4,4), kernel_initializer='he_uniform', activation='relu', padding='same', name='branch1_conv1')(x)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Concatenate(axis=-1)([c_1_bch, qp_input1])
    c_1_bch = Conv2D(16, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch1_conv2')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Conv2D(4, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch1_conv3')(c_1_bch) # Output should be 8x8x4, now start slicing.
    c_1_bch = BatchNormalization()(c_1_bch)


    c_1_pred = []
    c=0
    for i in range(8):
        for j in range(8):
            c+=1
            y = Lambda(lambda x, i,j: x[:, i,j,:],  arguments={'i': i,'j':j})(c_1_bch)
            c_1_pred.append(Activation('softmax',name='c1_predictions_' + str(c))(y))


    # block 2
    x = Conv2D(8, (3, 3),   kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3),  kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # branch 2
    c_2_bch = Conv2D(32, (4, 4), strides=(4, 4), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch2_conv1')(x)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Concatenate(axis=-1)([c_2_bch, qp_input2])
    c_2_bch = Conv2D(16, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch2_conv2')(c_2_bch)
    c_2_bch = BatchNormalization()(c_2_bch)
    c_2_bch = Conv2D(4, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch2_conv3')(c_2_bch)  # Output should be 4x4x4, now start slicing.
    c_2_bch = BatchNormalization()(c_2_bch)


    c_2_pred = []
    c = 0
    for i in range(4):
        for j in range(4):
            c += 1
            y = Lambda(lambda x, i,j: x[:, i, j,:],  arguments={'i': i,'j':j})(c_2_bch)
            c_2_pred.append(Activation('softmax',name='c2_predictions_' + str(c))(y))


    # block 3 
    x = Conv2D(16, (3, 3),   kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3),  kernel_initializer='he_uniform',  activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # branch 3
    c_3_bch = Conv2D(16, (4, 4), strides=(4, 4), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch3_conv1')(x)
    c_3_bch = BatchNormalization()(c_3_bch)
    c_3_bch = Concatenate(axis=-1)([c_3_bch, qp_input3])
    c_3_bch = Conv2D(8, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch3_conv2')(c_3_bch)
    c_3_bch = BatchNormalization()(c_3_bch)
    c_3_bch = Conv2D(4, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch3_conv3')(c_3_bch)  # Output should be 2x2x4, now start slicing.
    c_3_bch = BatchNormalization()(c_3_bch)
    c_3_pred=[]
    c = 0
    for i in range(2):
        for j in range(2):
            c += 1
            y = Lambda(lambda x, i,j: x[:, i, j,:],  arguments={'i': i,'j':j})(c_3_bch)
            c_3_pred.append(Activation('softmax', name='c3_predictions_' + str(c))(y))

    # block 4
    x = Conv2D(16, (3, 3),  kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3),  kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # branch 4
    c_4_bch = Conv2D(8, (4, 4), strides=(4, 4), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch4_conv1')(x)
    c_4_bch = BatchNormalization()(c_4_bch)
    c_4_bch = Concatenate(axis=-1)([c_4_bch, qp_input4])
    c_4_bch = Conv2D(4, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch4_conv2')(c_4_bch)
    c_4_bch = BatchNormalization()(c_4_bch)
    c_4_bch = Conv2D(4, (1, 1), kernel_initializer='he_uniform', activation='relu', padding='same',
                     name='branch4_conv3')(c_4_bch)  # Output should be 1x1x4, now start slicing.
    c_4_bch = BatchNormalization()(c_4_bch)

    y = Lambda(lambda x: x[:, 0,0, :])(c_4_bch)
    c_4_pred = Activation('softmax', name='c4_predictions')(y)


    preds=[]
    preds.append(c_4_pred)
    for i in range(4):
        preds.append(c_3_pred[i])

    for i in range(16):
        preds.append(c_2_pred[i])

    for i in range(64):
        preds.append(c_1_pred[i])

    model = Model(inputs=[img_input, qp_input1], outputs=preds, name='model')
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  # optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', dest='train_path', required=False, default='../Dummy_data/Training/',
                        help='Path to training data (default - Dummy_data/Training)')
    parser.add_argument('--val_path', dest='val_path', required=False, default='../Dummy_data/Validation/',
                        help='Path to validation data (default - Dummy_data/Validation)')
    parser.add_argument('--hist_path', dest='hist_path', required=False, default='logs_fcn',
                        help='Path to save the training and validation losses with training progress')
    parser.add_argument('--model_path', dest='model_path', required=False, default='Trained_model',
                        help='Path to save the trained model')
    args = parser.parse_args()
    X_train = glob.glob(args.train_path+"/*.png", recursive=True)

    X_train=np.asarray(X_train)
    np.random.shuffle(X_train)


    # print("Number of training samples\n",len(X_train))


    Y_train = np.asarray([w.replace('png', 'npz') for w in X_train])

    model=cnn_model()
    batch_size=128
    training_batch_generator = Data_Generator(X_train, Y_train, batch_size, shuffle=True)

    checkpointer = ModelCheckpoint(filepath=args.model_path+'/hfcn.hdf5', monitor='val_loss',
                                    verbose=1, save_weights_only=False, save_best_only=False)
    X_valid, qp1_valid, Y_valid = load_validation(args.val_path)
    steps=len(X_train) // batch_size
    # print("Steps per epoch",steps)
    model.fit_generator(load_data(np.array(X_train), np.array(Y_train), batch_size=128), epochs=15000, steps_per_epoch=steps, verbose=1, validation_data=([X_valid, qp1_valid], Y_valid),callbacks=[checkpointer, TrainValTensorBoard(write_graph=False, log_dir=args.hist_path)],  use_multiprocessing=True,workers=8,      max_queue_size=32)

