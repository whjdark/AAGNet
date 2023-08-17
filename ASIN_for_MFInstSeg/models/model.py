import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from models import  model_utils

def ASIN_model():
    input_points = layers.Input(shape=(model_utils.FaceNum, model_utils.PointNum, model_utils.PointFeatures))
    x = model_utils.Feature_Extraction_PointNet(input_points=input_points)
    x = layers.Reshape((model_utils.FaceNum, model_utils.FaceFeatures))(x)
    clsx = model_utils.Semantic_Segmentation(input_Faces=x)
    bfclsx = model_utils.Bottom_Face_Identification(input_Faces=x)

    y = model_utils.Encode_PointNet(input_Faces=x)
    s1 = model_utils.Dimension_Reduction(y)
    s1 = layers.Lambda(model_utils.Similarity_Calculate)(s1)
    s2 = layers.Lambda(model_utils.Similarity_Calculate)(y)
    sb1 = layers.BatchNormalization()(s1)
    sb2 = layers.BatchNormalization()(s2)
    sb1 = layers.Activation('sigmoid')(sb1)
    sb2 = layers.Activation('sigmoid')(sb2)
    s = (tf.add(sb1, sb2, name='instance_segmentation')) / 2.0

    MyModel = Model(inputs=input_points, outputs=[clsx, bfclsx, s])
    return MyModel

if __name__ ==  "__main__":
    FSRN_model = ASIN_model()
    FSRN_model.summary()