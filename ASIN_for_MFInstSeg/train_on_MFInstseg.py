import json
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from tensorflow.python.keras import backend as K
from models import model
from loss import Loss_Function
from acc import Instance_Segmentation_Accuracy


input_size = 128
num_classes = 26 # (24+1+1 for background and non-existing faces)

# Pathes to training data set and validation data set
data_path = 'E:\\traning_data\\data2'
data_partition_path = 'E:\\aagnet-upload\\MFInstseg_partition'

# Path to model weight
model_save_weights_file = 'models9\ASIN_weights.h5'
# Path to the best model weights
best_model_save_weithts_file = 'models9\ASIN_best_weights.h5'
# Path to the logging directory
logdir = 'logdir'


def load_points_OnFace(file_list):
    pc_path = os.path.join(data_path, 'pcs')
    data = []
    for i in range(len(file_list)):
        file_path = os.path.join(pc_path, file_list[i])
        file_path = file_path + '.npy'
        pc = np.load(file_path)
        # ps is (1, 128, 32, 6)
        data.append(pc)
    # stack to (n, 1, 128, 32, 6) and then squeeze to (n, 128, 32, 6)
    data = np.stack(data, axis=0)
    data = np.squeeze(data, axis=1)
    return data

def load_labels(file_list):
    label_path = os.path.join(data_path, 'labels')
    # load json
    seg_labels = []
    inst_labels = []
    bottom_labels = []
    for i in range(len(file_list)):
        file_path = os.path.join(label_path, file_list[i])
        file_path = file_path + '.json'
        with open(str(file_path), "r") as read_file:
            labels_data = json.load(read_file)
        _, labels = labels_data[0]
        seg_label, inst_label, bottom_label = labels['seg'], labels['inst'], labels['bottom']
        # read semantic segmentation label for each face
        num_faces = len(seg_label)
        face_segmentaion_labels = np.zeros((input_size, num_classes), dtype=np.int32) # pad to input_size
        for idx, face_id in enumerate(range(num_faces)):
            index = seg_label[str(face_id)]
            face_segmentaion_labels[idx][index] = 1 # one-hot
            # stock face is 24
        # missing face is 25
        for idx in range(num_faces, 128):
            face_segmentaion_labels[idx][25] = 1 # one-hot
        # print(file_path)
        # print(face_segmentaion_labels.shape)
        # for i in range(face_segmentaion_labels.shape[0]):
        #     print(i, face_segmentaion_labels[i])

        # read instance segmentation labels for each instance
        # just a face adjacency
        instance_label = np.array(inst_label, dtype=np.int32)
        # pad to input_size
        identity = np.ones((input_size - num_faces, input_size - num_faces), dtype=np.int32)
        pad_mat1 = np.zeros((num_faces, input_size - num_faces), dtype=np.int32)
        pad_mat2 = np.zeros((input_size - num_faces, num_faces), dtype=np.int32)
        instance_label = np.block([[instance_label, pad_mat1], [pad_mat2, identity]])
        # print(file_path)
        # print(instance_label.shape)
        # for i in range(instance_label.shape[0]):
        #     print(instance_label[i])
        
        # read bottom face segmentation label for each face 
        bottom_segmentaion_labels = np.zeros(input_size) # pad to input_size
        for idx, face_id in enumerate(range(num_faces)):
            index = bottom_label[str(face_id)]
            bottom_segmentaion_labels[idx] = index
        bottom_segmentaion_labels = bottom_segmentaion_labels.reshape(-1, 1)
        # print(file_path)
        # print(bottom_segmentaion_labels.shape)
        # print(bottom_segmentaion_labels)

        # stack to (n, 128)
        seg_labels.append(face_segmentaion_labels)
        inst_labels.append(instance_label)
        bottom_labels.append(bottom_segmentaion_labels)
    seg_labels = np.stack(seg_labels, axis=0)
    inst_labels = np.stack(inst_labels, axis=0)
    bottom_labels = np.stack(bottom_labels, axis=0)
    return seg_labels, bottom_labels, inst_labels


if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()

    with open(os.path.join(data_partition_path, 'train.txt'), 'r') as f:
        train_filelist = f.readlines()
        train_filelist = [x.strip() for x in train_filelist]

    with open(os.path.join(data_partition_path, 'val.txt'), 'r') as f:
        validation_filelist = f.readlines()
        validation_filelist = [x.strip() for x in validation_filelist]

    train_points_OnFace = load_points_OnFace(train_filelist)
    print(train_points_OnFace.shape)
    train_labels = load_labels(train_filelist)
    train_seg_labels = train_labels[0]
    train_bottom_labels = train_labels[1]
    train_similar_matrix = train_labels[2]
    print(train_seg_labels.shape)
    print(train_similar_matrix.shape)
    print(train_bottom_labels.shape)

    validation_points_OnFace = load_points_OnFace(validation_filelist)
    print(validation_points_OnFace.shape)
    validation_labels = load_labels(validation_filelist)
    validation_seg_labels = validation_labels[0]
    validation_bottom_labels = validation_labels[1]
    validation_similar_matrix = validation_labels[2]
    print(validation_seg_labels.shape)
    print(validation_similar_matrix.shape)
    print(validation_bottom_labels.shape)

    mymodel = model.ASIN_model()
    mymodel.summary()

    # Decrease the learning rate
    def scheduler(epoch):
        if epoch < 60:
            K.set_value(mymodel.optimizer.lr, 0.001)
        else:
            if (epoch - 60) % 15 == 0 or epoch == 60:
                lr = K.get_value(mymodel.optimizer.lr)
                K.set_value(mymodel.optimizer.lr, lr * 0.5)
                print("lr changed to {}".format(lr * 0.5))
        return K.get_value(mymodel.optimizer.lr)
    reduce_lr = LearningRateScheduler(scheduler)

    optimizer_adam = optimizers.Adam(lr=0.001)
    mymodel.compile(optimizer=optimizer_adam,
                    loss=[losses.categorical_crossentropy,
                        losses.binary_crossentropy,
                        Loss_Function],
                    loss_weights=[1, 1, 10],
                    # metrics={'segment':[Instance_Segmentation_Accuracy],
                    #          'segmentbottom':['accuracy'],
                    #          'tf.math.truediv':['accuracy']},
                    metrics=[Instance_Segmentation_Accuracy]
                    )

    time_start=time.time()

    with tf.device('/GPU:0'):
        tbcallbacktrain = TensorBoard(log_dir=logdir)
        checkpoint = ModelCheckpoint(best_model_save_weithts_file, monitor='val_loss', save_best_only=True, mode='auto',
                                    period=1)
        scoretrain = mymodel.fit(train_points_OnFace,
                                [train_labels, train_bottom_labels, train_similar_matrix],
                                batch_size=64, epochs=120, shuffle=True,
                                validation_data=(validation_points_OnFace, [validation_labels, validation_bottom_labels, validation_similar_matrix]),
                                callbacks=[tbcallbacktrain, reduce_lr, checkpoint],
                                verbose=2)
        mymodel.save_weights(model_save_weights_file)
        time_end = time.time()
    print('totally cost', time_end - time_start)