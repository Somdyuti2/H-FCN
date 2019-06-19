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

import keras.backend as K
from keras.layers import Softmax, Reshape, Concatenate, Lambda, Input, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam

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

    # --- block 1 ---
    x = Conv2D(8, (3, 3), kernel_initializer='he_normal', activation='relu', padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3),  kernel_initializer='he_normal', activation='relu', padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) #output size 32 x32 x16

    # # --- coarse 1 branch ---
    c_1_bch = Conv2D(32, (4, 4), strides=(4,4), kernel_initializer='he_normal', activation='relu', padding='same', name='branch1_conv1')(x)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Concatenate(axis=-1)([c_1_bch, qp_input1])
    c_1_bch = Conv2D(16, (1, 1), kernel_initializer='he_normal', activation='relu', padding='same',
                     name='branch1_conv2')(c_1_bch)
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch = Conv2D(4, (1, 1), kernel_initializer='he_normal', activation='relu', padding='same',
                     name='branch1_conv3')(c_1_bch) # Output should be 8x8x4, now start slicing.
    c_1_bch = BatchNormalization()(c_1_bch)
    c_1_bch=Softmax(name='c1_softmax', axis=-1)(c_1_bch)
    c_1_pred=Reshape([64,4])(c_1_bch)
    # c_1_pred=Activation('linear',name='c1_predictions')(c_1_pred)

    # --- block 2 ---
    x = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # # --- coarse 2 branch ---
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
    c_2_bch = Softmax(name='c2_softmax', axis=-1)(c_2_bch)
    c_2_pred = Reshape([16, 4])(c_2_bch)

    # --- block 3 ---
    x = Conv2D(16, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # --- coarse 3 branch ---
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
    c_3_bch = Softmax(name='c3_softmax', axis=-1)(c_3_bch)
    c_3_pred = Reshape([4, 4])(c_3_bch)

    # --- block 4 ---
    x = Conv2D(16, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), kernel_initializer='he_uniform', activation='relu', padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # # --- coarse 4 branch ---
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
    c_4_bch = Softmax(name='c4_softmax', axis=-1)(c_4_bch)
    c_4_pred = Reshape([1, 4])(c_4_bch)

    preds=[]
    preds.append(c_4_pred)
    preds.append(c_3_pred)
    preds.append(c_2_pred)
    preds.append(c_1_pred)


    model = Model(inputs=[img_input, qp_input1], outputs=preds, name='model')
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.summary()
    return model