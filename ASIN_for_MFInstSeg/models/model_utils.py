import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

FaceNum = 128  # The number of faces extracted from each model
PointNum = 32 # The number of points extracted from each face
PointFeatures = 6  # The information dimension of a point
FaceFeatures = 32  # The information dimension of a face
Classnum = 25 # The number of feature types 24 + no-exist face

# expend dimension
def exp_dim(global_feature, num_points):
    return tf.tile(global_feature,
                   [1, num_points, 1])

# Feature extraction
def Feature_Extraction_PointNet(num_points=PointNum, in_numoffeatures=PointFeatures, out_numoffeatures=FaceFeatures, numoffaces=FaceNum
               , input_points=layers.Input(shape=(FaceNum, PointNum, PointFeatures))):
    x = layers.Convolution2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu',
                      input_shape=(numoffaces, num_points, in_numoffeatures))(input_points)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(numoffaces, num_points))(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(in_numoffeatures * in_numoffeatures, weights=[np.zeros([256, in_numoffeatures * in_numoffeatures]),
                                                            np.eye(in_numoffeatures).flatten().astype(np.float32)])(x)
    input_T = layers.Reshape((in_numoffeatures, in_numoffeatures))(x)

    input_Reshape = layers.Reshape((numoffaces * num_points, in_numoffeatures))(input_points)
    g = tf.matmul(input_Reshape, input_T)
    g = layers.Reshape((numoffaces, num_points, in_numoffeatures))(g)
    g = layers.Convolution2D(filters=64, kernel_size=(1, 1), strides=(1, 1), input_shape=(numoffaces,
                                                                                   num_points, in_numoffeatures),
                      activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(g)
    g = layers.BatchNormalization()(g)

    f = layers.Convolution2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(g)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling2D(pool_size=(numoffaces, num_points))(f)
    f = layers.Dense(512, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = layers.Reshape((64, 64))(f)

    g = layers.Reshape((numoffaces * num_points, 64))(g)
    g = tf.matmul(g, feature_T)
    g = layers.Reshape((numoffaces, num_points, 64))(g)
    g = layers.Convolution2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution2D(filters=128, kernel_size=(1, 1), strides=(1, 1), activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution2D(filters=1024, kernel_size=(1, 1), strides=(1, 1), activation='relu')(g)
    g = layers.BatchNormalization()(g)

    global_feature = layers.MaxPooling2D(pool_size=(1, num_points))(g)

    c = layers.Convolution2D(filters=512, kernel_size=(1, 1), strides=(1, 1), activation='relu')(global_feature)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution2D(filters=256, kernel_size=(1, 1), strides=(1, 1), activation='relu')(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution2D(filters=out_numoffeatures, kernel_size=(1, 1), strides=(1, 1), activation='relu')(c)
    c = layers.BatchNormalization(name="OUTPUT1-32")(c)
    prediction = c
    return prediction

#Semantic segmentation
def Semantic_Segmentation(num_Faces=FaceNum, numofclasses=Classnum + 1, input_Faces=layers.Input(shape=(FaceNum, FaceFeatures))):
    g = layers.Convolution1D(64, 1, input_shape=(num_Faces, FaceFeatures))(input_Faces)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    seg_part1 = g
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(128, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(1024, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    global_feature = layers.MaxPooling1D(pool_size=num_Faces)(g)
    global_feature = layers.Lambda(exp_dim, arguments={'num_points': num_Faces})(global_feature)

    c = layers.concatenate([seg_part1, global_feature])
    c = layers.Convolution1D(512, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(256, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(128, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    prediction = layers.Convolution1D(numofclasses, 1, activation='softmax', name='segment')(c)

    return prediction

#Bottom face identification
def Bottom_Face_Identification(num_Faces = FaceNum, input_Faces = layers.Input(shape=(FaceNum, FaceFeatures))):
    g = layers.Convolution1D(64, 1, input_shape=(num_Faces, FaceFeatures))(input_Faces)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    seg_part1 = g
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(128, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(1024, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    global_feature = layers.MaxPooling1D(pool_size=num_Faces)(g)
    global_feature = layers.Lambda(exp_dim, arguments={'num_points': num_Faces})(global_feature)

    c = layers.concatenate([seg_part1, global_feature])
    c = layers.Convolution1D(512, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(256, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(128, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.BatchNormalization()(c)
    prediction = layers.Convolution1D(1, 1, activation='sigmoid', name='segmentbottom')(c)

    return prediction

#Feature Encode
def Encode_PointNet(num_Faces=FaceNum, input_Faces=layers.Input(shape=(FaceNum, FaceFeatures))):
    g = layers.Convolution1D(64, 1, input_shape=(num_Faces, FaceFeatures))(input_Faces)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    seg_part1 = g
    g = layers.Convolution1D(64, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(128, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(1024, 1)(g)
    g = layers.LeakyReLU(alpha=0.3)(g)
    g = layers.BatchNormalization()(g)

    global_feature = layers.MaxPooling1D(pool_size=num_Faces)(g)
    global_feature = layers.Lambda(exp_dim, arguments={'num_points': num_Faces})(global_feature)

    c = layers.concatenate([seg_part1, global_feature])
    c = layers.Convolution1D(512, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.Dropout(0.2)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(256, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.Dropout(0.2)(c)
    c = layers.BatchNormalization()(c)
    c = layers.Convolution1D(64, 1)(c)
    c = layers.LeakyReLU(alpha=0.3)(c)
    c = layers.Dropout(0.2)(c)
    c = layers.BatchNormalization(name="OUTPUT2-64")(c)
    prediction = c
    return prediction

#Face similarity matrix calculate
def Similarity_Calculate(FacesMatrix):
    n = FacesMatrix.shape[1]
    FM1 = tf.expand_dims(FacesMatrix, axis=3)
    FM2 = tf.transpose(FacesMatrix, perm=[0, 2, 1])
    FM2 = tf.expand_dims(FM2, axis=1)
    FM1 = tf.tile(FM1, [1, 1, 1, n])
    FM2 = tf.tile(FM2, [1, n, 1, 1])
    SimilarityMatrix = tf.square(FM1 - FM2)
    SimilarityMatrix = -tf.reduce_sum(SimilarityMatrix, axis=2)
    return SimilarityMatrix

#Dimension reduction
def Dimension_Reduction(input=layers.Input(shape=(FaceNum, FaceFeatures))):
    x = layers.Convolution1D(64, 1)(input)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(32, 1)(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(1, 1)(x)
    x = layers.LeakyReLU(alpha=0.3)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.BatchNormalization(name="OUTPUT2-1")(x)
    return x