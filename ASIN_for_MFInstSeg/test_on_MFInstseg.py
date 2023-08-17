import os
import h5py
import numpy as np
import json
from models import model
from sklearn.cluster import estimate_bandwidth
import tensorflow as tf
from tensorflow.python.keras import backend as K
from models import cluster, model
import torch
from tqdm import tqdm


from torchmetrics.classification import (
    MulticlassAccuracy, 
    BinaryAccuracy, 
    BinaryF1Score, 
    BinaryJaccardIndex, 
    MulticlassJaccardIndex,
    BinaryAveragePrecision)



input_size = 128
num_classes = 26 # (24+1+1 for background and non-existing faces)
EPS = 1E-6
# Pathes to training data set and validation data set
data_path = 'E:\\traning_data\\data2'
data_partition_path = 'E:\\aagnet-upload\\MFInstseg_partition'
feat_names = ['chamfer', 'through_hole', 'triangular_passage', 'rectangular_passage', '6sides_passage',
              'triangular_through_slot', 'rectangular_through_slot', 'circular_through_slot',
              'rectangular_through_step', '2sides_through_step', 'slanted_through_step', 'Oring', 'blind_hole',
              'triangular_pocket', 'rectangular_pocket', '6sides_pocket', 'circular_end_pocket',
              'rectangular_blind_slot', 'v_circular_end_blind_slot', 'h_circular_end_blind_slot',
              'triangular_blind_step', 'circular_blind_step', 'rectangular_blind_step', 'round', 'stock'
             ]
# Path to model weights
model_save_weights_file = 'models6\\ASIN_best_weights.h5'




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
    real_num_faces = []
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
        #     print(face_segmentaion_labels[i])

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
        real_num_faces.append(num_faces)
    
    seg_labels = np.stack(seg_labels, axis=0)
    inst_labels = np.stack(inst_labels, axis=0)
    bottom_labels = np.stack(bottom_labels, axis=0)
    real_num_faces = np.stack(real_num_faces, axis=0)
    return seg_labels, bottom_labels, inst_labels, real_num_faces


def print_class_metric(metric):
    string = ''
    for i in range(len(metric)):
        string += feat_names[i] + ': ' + str(metric[i]) + ', '
    print(string)


class FeatureInstance():
    def __init__(self, name:int = None, 
                       faces:np.array = None, 
                       bottoms:list = None):
        self.name = name
        self.faces = faces
        self.bottoms = bottoms


def parser_label(inst_label, seg_label, bottom_label):
    label_list = []
    # parse instance label
    inst_label = np.array(inst_label, dtype=np.uint8)
    used_faces = []
    for row_idx, row in enumerate(inst_label):
        if np.sum(row) == 0:
            # stock face, no linked face, so the sum of the column is 0
            continue
        # when I_ij = 1 mean face_i is linked with face_j
        # so can get the indices of linked faces in a instance
        linked_face_idxs = np.where(row==1)[0]
        # used
        if len(set(linked_face_idxs).intersection(set(used_faces))) > 0:
            # the face has been assigned to a instance
            continue
        # create new feature
        new_feat = FeatureInstance()
        new_feat.faces = linked_face_idxs
        used_faces.extend(linked_face_idxs)
        # all the linked faces in a instance 
        # have the same segmentation label
        # so get the name of the instance
        a_face_id = new_feat.faces[0]
        seg_id = seg_label[int(a_face_id)]
        new_feat.name = seg_id
        # get the bottom face segmentation label
        # new_feat.bottoms = np.where(bottom_label==1)[0]
        # # add new feature into list and used face counter
        label_list.append(new_feat)
    
    return label_list


# def post_process(out, inst_thres, bottom_thres):
#     seg_out, inst_out, bottom_out = out
#     # post-processing for semantic segmentation 
#     # face_logits = torch.argmax(seg_out, dim=1)
#     face_logits = seg_out
#     # post-processing for instance segmentation 

#     adj = inst_out > inst_thres
#     adj = adj.astype('int32')

#     # post-processing for bottom classification 
#     # bottom_out = sigmoid(bottom_out)
#     # bottom_logits = bottom_out > bottom_thres
#     # bottom_logits = bottom_logits
    
#     # Identify individual proposals of each feature
#     proposals = set() # use to delete repeat proposals
#     # record whether the face belongs to a instance
#     used_flags = np.zeros(adj.shape[0], dtype=np.bool_)
#     for row_idx, row in enumerate(adj):
#         if used_flags[row_idx]:
#             # the face has been assigned to a instance
#             continue
#         if np.sum(row) <= EPS: 
#             # stock face, no linked face, so the sum of the column is 0
#             continue
#         # non-stock face
#         proposal = set() # use to delete repeat faces
#         for col_idx, item in enumerate(row):
#             if used_flags[col_idx]:
#                 # the face has been assigned to a proposal
#                 continue
#             if item: # have connections with currect face
#                 proposal.add(col_idx)
#                 used_flags[col_idx] = True
#         if len(proposal) > 0:
#             proposals.add(frozenset(proposal)) # frozenset is a hashable set

#     # print(proposals)
#     # TODO: better post-process
    
#     # save new results
#     features_list = []
#     for instance in proposals:
#         instance = list(instance)
#         # sum voting for the class of the instance
#         sum_inst_logit = 0
#         for face in instance:
#             sum_inst_logit += face_logits[face]
#         # the index of max score is the class of the instance
#         inst_logit = np.argmax(sum_inst_logit)
#         if inst_logit == 24 or inst_logit == 25:
#             # is stock or non-existing face, ignore
#             continue
#         # get instance label name from face_logits

#         inst_name = inst_logit
#         # get the bottom faces
#         # bottom_faces = []
#         # for face_idx in instance:
#         #     if bottom_logits[face_idx]:
#         #         bottom_faces.append(face_idx)
#         features_list.append(
#             FeatureInstance(name=inst_name, faces=np.array(instance)))
    
#     return features_list


def post_process2(r_instances, r_segments, r_bottomface, num_real_faces, PartResultE, PartResultD):
    '''
    supppot for instance cluster, which can significantly improve per-feature perfmance
    '''
    AllclustersResults = []
    # ClusterResults = []
    # ClusterSimilarityMatrix = []
    ModifyPredictResult = []
    predict_FacesId = np.array([id for id in range(num_real_faces)], dtype=int)

    NewPartResultD = PartResultD
    for j in range(PartResultE.shape[1]):
        if j != 0:
            PartResultD = np.hstack((PartResultD, NewPartResultD))
    #PartResult = np.hstack((PartResultE, PartResultD))
    PartResult = (PartResultE+PartResultD)/2.0
    mycluster = cluster.ClusterMethod(PartResult)

    # Predict the bandwidth by adjusting quantile.
    # The value of quantile can be 0.14, 0.15, or 0.16, which may lead to a better result.

    bandwidth = estimate_bandwidth(PartResult, quantile=0.16)
    pred_gmm = mycluster.MeanShift(bandwidth)
    TrueFaceId = np.array(predict_FacesId, dtype=int)
    PredictSimilarMatrix = r_instances

    PredictSimilarMatrixINT = np.eye(PredictSimilarMatrix.shape[0])
    for j in range(PredictSimilarMatrix.shape[0]):
        for k in range(PredictSimilarMatrix.shape[1]):
            if PredictSimilarMatrix[j][k] > 0.5:
                PredictSimilarMatrixINT[j][k] = 1
                PredictSimilarMatrixINT[k][j] = 1

    ClusterResult = []
    CRClasses = []
    for j in range(len(pred_gmm)):
        if j == 0:
            CRClasses.append(pred_gmm[j])
        else:
            if pred_gmm[j] not in CRClasses:
                CRClasses.append(pred_gmm[j])
    for j in range(len(CRClasses)):
        CR = []
        CRClass = CRClasses[j]
        for k in range(len(pred_gmm)):
            if CRClass == pred_gmm[k]:
                CR.append(TrueFaceId[k])
        ClusterResult.append(CR)
    AllclustersResults.append(ClusterResult)
    SimilarityMatrix = np.eye(len(pred_gmm))
    allindexes = []
    for Facesinoneinstance in ClusterResult:
        if len(Facesinoneinstance) > 1:
            indexes = []
            for Faceinoneinstance in Facesinoneinstance:
                FaceinoneinstanceINT = int(Faceinoneinstance)
                TrueFaceIdList = TrueFaceId.tolist()
                indexes.append(TrueFaceIdList.index(Faceinoneinstance))
            for j in range(len(indexes)):
                for k in range(j, len(indexes)):
                    SimilarityMatrix[indexes[j]][indexes[k]] = 1
                    SimilarityMatrix[indexes[k]][indexes[j]] = 1
            zeroid = []
            for j in range(len(TrueFaceId)):
                if TrueFaceId[j] == 0:
                    zeroid.append(j)
            for j in range(len(zeroid)):
                for k in range(j, len(zeroid)):
                    SimilarityMatrix[zeroid[j]][zeroid[k]] = 1
                    SimilarityMatrix[zeroid[k]][zeroid[j]] = 1
    NewPredictMatrix = np.zeros((SimilarityMatrix.shape[0], SimilarityMatrix.shape[1]))
    for j in range(SimilarityMatrix.shape[0]):
        for k in range(SimilarityMatrix.shape[1]):
            if (PredictSimilarMatrixINT[j][k] == 1) & (SimilarityMatrix[j][k] == 1):
                NewPredictMatrix[j][k] = 1

    # ClusterSimilarityMatrix = SimilarityMatrix
    ModifyPredictResult = NewPredictMatrix

    # point2_time = time.time()
    # print('point2 time:', point2_time-point1_time)

    # PredictResult_Instances = []
    TrueFaceId = predict_FacesId
    SimilarMatrix = ModifyPredictResult
    # SimilarMatrixINT = np.round(SimilarMatrix)
    allindexsim = []
    for j in range(SimilarMatrix.shape[0]):
        indexsim = []
        # flag = 0
        for k in range(SimilarMatrix.shape[1]):
            if round(SimilarMatrix[j][k]) == 1:
                indexsim.append(k)
        if len(indexsim)>0:
            allindexsim.append(indexsim)
    instance_results=[]

    # keypoint1_time = time.time()
    # print('keypoint1 time:', keypoint1_time-point2_time)

    allindexofallindexsim = []
    MaxScores = []
    for j in range(SimilarMatrix.shape[1]):
        indexofallindexsim = []
        for myindexsim in allindexsim:
            if j in myindexsim:
                indexofallindexsim.append(myindexsim)
        allindexofallindexsim.append(indexofallindexsim)
    
    # keypoint2_time = time.time()
    # print('keypoint2 time:', keypoint2_time-keypoint1_time)

    allscoreofproposal = []
    MaxScoreFaceProposal = []
    yi = 0
    for myindexofallindexsim in allindexofallindexsim:
        scoreinonepropose = []
        for eveyindexes in myindexofallindexsim:
            sumscore = 0
            for eveyindex in eveyindexes:
                if yi != eveyindex:
                    sumscore += SimilarMatrix[yi][eveyindex]
            avescore = sumscore / (len(eveyindexes))
            scoreinonepropose.append(avescore)
        yi += 1
        maxscore = max(scoreinonepropose)
        MaxScores.append(maxscore)
        indexofMaxScore = scoreinonepropose.index(maxscore)
        MaxScoreFaceProposal.append(myindexofallindexsim[indexofMaxScore])
        allscoreofproposal.append(scoreinonepropose)

    DiffentGroups = []
    New_MaxScores = [] 
    for j in range(len(MaxScoreFaceProposal)):
        if j == 0:
            DiffentGroups.append(MaxScoreFaceProposal[j])
            New_MaxScores.append(MaxScores[j])

        else:
            if MaxScoreFaceProposal[j] not in DiffentGroups:
                DiffentGroups.append(MaxScoreFaceProposal[j])
                New_MaxScores.append(MaxScores[j]) 

    # point4_time = time.time()
    # print('point4 time:', point4_time-point3_time)

    IsolatedProposals = []
    IntersectProposals = []
    IntersectScores = []
    kk = 0 
    for diffentgroup in DiffentGroups: 
        num = 0
        for element in diffentgroup:
            for otherdiffentgroup in DiffentGroups:
                if diffentgroup != otherdiffentgroup:
                    if element in otherdiffentgroup:
                        num += 1
                        break
        if num == 0:
            IsolatedProposals.append(diffentgroup)
        else:
            IntersectProposals.append(diffentgroup)
            IntersectScores.append(New_MaxScores[kk])
        kk += 1

    indexesofIntersectScores = sorted(range(len(IntersectScores)), key=lambda k: IntersectScores[k], reverse=True)

    # point5_time = time.time()
    # print('point5 time:', point5_time-point4_time)
    
    kk = 0
    for index in indexesofIntersectScores:
        num = 0
        if kk == 0:
            IsolatedProposals.append(IntersectProposals[index])
        else:
            for proposal in IsolatedProposals:
                for k in range(len(IntersectProposals[index])):
                    if IntersectProposals[index] != proposal:
                        if IntersectProposals[index][k] in proposal:
                            num = 1
                            break
                if num == 1:
                    break
            if num == 0:
                if IntersectProposals[index] not in IsolatedProposals:
                    IsolatedProposals.append(IntersectProposals[index])
        kk += 1

    # point6_time = time.time()
    # print('point6 time:', point6_time-point5_time)

    for myindexsim in IsolatedProposals:
        instance_result = []
        for myindex in myindexsim:
            instance_result.append(TrueFaceId[myindex])
        instance_results.append(instance_result)

    # print(predict_FacesId.shape)
    # print(r_segments.shape)
    # print(r_bottomface.shape)
    # print(PredictResult_Instances)
    
    # save new results
    features_list = []
    for instance in instance_results:
        instance = list(instance)
        # sum voting for the class of the instance
        sum_inst_logit = 0
        for face in instance:
            sum_inst_logit += r_segments[face]
        # the index of max score is the class of the instance
        inst_logit = np.argmax(sum_inst_logit)
        if inst_logit == 24 or inst_logit == 25:
            # is stock or non-existing face, ignore
            continue
        # get instance label name from face_logits

        inst_name = inst_logit
        # get the bottom faces
        # bottom_faces = []
        # for face_idx in instance:
        #     if bottom_logits[face_idx]:
        #         bottom_faces.append(face_idx)
        features_list.append(
            FeatureInstance(name=inst_name, faces=np.array(instance)))
    
    return features_list


def cal_recognition_performance(feature_list, label_list):
    # one hot encoding
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        pred[feature.name] += 1
    for label in label_list:
        gt[label.name] += 1
    tp = np.minimum(gt, pred)

    return pred, gt, tp


def cal_localization_performance(feature_list, label_list):
    # one hot encoding
    pred = np.zeros(24, dtype=int)
    gt = np.zeros(24, dtype=int)
    for feature in feature_list:
        pred[feature.name] += 1
    for label in label_list:
        gt[label.name] += 1
    
    # sort the feature_list and label_list by name
    feature_list.sort(key=lambda x: x.name)
    label_list.sort(key=lambda x: x.name)
    tp = np.zeros(24, dtype=int)

    
    found_lbl = np.zeros(len(label_list))
    # for each detection
    for pred_i in range(len(feature_list)):
        pred_name = feature_list[pred_i].name

        #among the ground-truths, choose one that belongs to the same class and has the highest IoU with the detection        
        for lbl_i in range(len(label_list)):  
            lbl_name = label_list[lbl_i].name
        
            if pred_name != lbl_name or found_lbl[lbl_i] == 1:
                    continue
            
            # compute IoU
            pred_faces = feature_list[pred_i].faces
            lbl_faces = label_list[lbl_i].faces
            intersection = np.intersect1d(pred_faces, lbl_faces)
            union = np.union1d(pred_faces, lbl_faces)
            iou = len(intersection) / len(union)

            # when IOU == 1, the detection is correct
            # else the detection is wrong
            if iou >= 1.0 - EPS:
                found_lbl[lbl_i] = 1
                tp[pred_name] += 1
                break
    
    # when tp gt not equal, print the detail
    # if not np.all(tp == gt):
    #     for feature in feature_list:
    #         feature.faces.sort()
    #         print('feature', feature.name, feature.faces)
    #     for label in label_list:
    #         label.faces.sort()
    #         print('label', label.name, label.faces)

    #     print('tp', tp)
    #     print('pd', pred)
    #     print('gt', gt)

    return pred, gt, tp


def eval_metric(pre, trul, tp):
    precision = tp / pre
    recall = tp / trul
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    # if the gt[i] == 0, mean class i is not in the ground truth
    # so the precision and recall of class i is not defined
    # so set the precision and recall of class i to 1
    precision[trul == 0] = 1
    recall[trul == 0] = 1

    return precision, recall


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    with open(os.path.join(data_partition_path, 'test.txt'), 'r') as f:
        test_filelist = f.readlines()
        test_filelist = [x.strip() for x in test_filelist]

    test_points_OnFace = load_points_OnFace(test_filelist)
    print(test_points_OnFace.shape)
    test_labels = load_labels(test_filelist)
    test_seg_labels = test_labels[0]
    test_bottom_labels = test_labels[1]
    test_similar_matrix = test_labels[2]
    real_num_faces = test_labels[3]
    print(test_seg_labels.shape)
    print(test_similar_matrix.shape)
    print(test_bottom_labels.shape)

    loaded_model = model.ASIN_model()
    # loaded_model.summary()
    loaded_model.load_weights(model_save_weights_file)
    

    # r_Feature_Recogintion = np.round(results[0])
    # r_Bottom_Face_Recogintion = np.round(results[1])
    # r_Instances_Segment = np.round(results[2])

    #time_end = time.time()
    
    # k_FR = 0
    # for i in range(r_Feature_Recogintion.shape[0]):
    #     for j in range(r_Feature_Recogintion.shape[1]):
    #         tt = np.array(r_Feature_Recogintion[i, j, :], dtype=np.int)
    #         tt2 = train_seg_labels[i, j, :]
    #         if all(tt == tt2):
    #             k_FR  += 1
    # acc_Feature_Recogintion = k_FR / (r_Feature_Recogintion.shape[0]*r_Feature_Recogintion.shape[1])
    # print(acc_Feature_Recogintion)

    # k_BFR = 0
    # for i in range(r_Bottom_Face_Recogintion.shape[0]):
    #     for j in range(r_Bottom_Face_Recogintion.shape[1]):
    #         tt = np.array(r_Bottom_Face_Recogintion[i, j, :], dtype=np.int)
    #         tt2 = train_bottom_labels[i, j, :]
    #         if all(tt == tt2):
    #             k_BFR += 1
    # acc_Feature_Recogintion_Bottom = k_BFR / (r_Bottom_Face_Recogintion.shape[0]*r_Bottom_Face_Recogintion.shape[1])
    # print(acc_Feature_Recogintion_Bottom)

    # k_IS = 0
    # for i in range(r_Instances_Segment.shape[0]):
    #     for j in range(r_Instances_Segment.shape[1]):
    #         tt = np.array(r_Instances_Segment[i, j, :], dtype=np.int)
    #         tt2 = train_similar_matrix[i, j, :]
    #         if all(tt == tt2):
    #             k_IS += 1
    # acc_Instances_Segment = k_IS / (r_Instances_Segment.shape[0]*r_Instances_Segment.shape[1])
    # print(acc_Instances_Segment)
    
    # ----------------------per-face metric----------------------
    test_seg_acc = MulticlassAccuracy(num_classes=26)
    test_inst_acc = BinaryAccuracy()
    test_bottom_acc = BinaryAccuracy()
    
    test_seg_iou = MulticlassJaccardIndex(num_classes=26)
    test_inst_f1 = BinaryF1Score()
    # test_inst_ap = BinaryAveragePrecision().to(device)
    test_bottom_iou = BinaryJaccardIndex()

    # ----------------------per-instance metric----------------------
    rec_predictions = np.zeros(24, dtype=int)
    rec_truelabels = np.zeros(24, dtype=int)
    rec_truepositives = np.zeros(24, dtype=int)

    loc_predictions = np.zeros(24, dtype=int)
    loc_truelabels = np.zeros(24, dtype=int)
    loc_truepositives = np.zeros(24, dtype=int)

    for i in tqdm(range(test_points_OnFace.shape[0])):
        input = np.expand_dims(test_points_OnFace[i], 0)
        with tf.device('/GPU:0'):
            results = loaded_model.predict(input)
        r_Feature_Recogintion = results[0][0]
        r_Bottom_Face_Recogintion = results[1][0]
        r_Instances_Segment = results[2][0]

        # print(r_Feature_Recogintion.shape, r_Bottom_Face_Recogintion.shape, r_Instances_Segment.shape)

        deduction_layers = loaded_model.get_layer('OUTPUT2-1')
        embedding_layers = loaded_model.get_layer('OUTPUT2-64')
        input_layers = loaded_model.input
        iterate = K.function([input_layers], [embedding_layers.output[0]])
        iterate2 = K.function([input_layers], [deduction_layers.output[0]])
        embedding_layers_output = iterate([input])
        deduction_layers_output = iterate2([input])

        # print(embedding_layers_output[0].shape, deduction_layers_output[0].shape)

        # per-face metric
        # THE NON-EXISTING FACES SHOULD NOT BE INCLUDED
        real_num_f = real_num_faces[i]
        for j in range(real_num_f):
            seg_label = np.argmax(test_seg_labels[i, j, :])
            if seg_label == 25:
                # ignore non-existing faces
                continue
            # seg 
            seg_pred = np.array(r_Feature_Recogintion[j, :], dtype=np.float)
            seg_pred = torch.from_numpy(seg_pred).unsqueeze(0)
            seg_label = np.array([seg_label])
            seg_label = torch.from_numpy(seg_label)
            # print('----------------test_seg_labels------------------------------')
            # print(seg_pred.argmax(), seg_label)
            test_seg_acc.update(seg_pred, seg_label)
            test_seg_iou.update(seg_pred, seg_label)
            seg_pred = np.argmax(seg_pred)

            # inst
            inst_pred = np.array(r_Instances_Segment[j, :], dtype=np.float)
            inst_pred = torch.from_numpy(inst_pred).unsqueeze(0)
            inst_label = np.array(test_similar_matrix[i, j, :], dtype=np.int)
            inst_label = torch.from_numpy(inst_label)
            inst_label = inst_label.unsqueeze(0)
            # print('----------------inst_pred------------------------------')
            # print(inst_pred > 0.5)
            # print(inst_label.bool())
            test_inst_acc.update(inst_pred, inst_label)
            test_inst_f1.update(inst_pred, inst_label)

            # bottom
            bottom_pred = np.array(r_Bottom_Face_Recogintion[j, :], dtype=np.float)
            bottom_pred = torch.from_numpy(bottom_pred).unsqueeze(0)
            bottom_label = np.array(test_bottom_labels[i, j, :], dtype=np.int)
            bottom_label = torch.from_numpy(bottom_label)
            bottom_label = bottom_label.unsqueeze(0)
            # print('----------------bottom_label------------------------------')
            # print(bottom_pred, bottom_label)
            test_bottom_acc.update(bottom_pred, bottom_label)
            test_bottom_iou.update(bottom_pred, bottom_label)
        
        # per-instance metric
        seg_out = np.array(r_Feature_Recogintion, dtype=np.float)
        seg_out = seg_out[:real_num_f] # padding faces are discarded

        inst_out = np.array(r_Instances_Segment, dtype=np.float)
        inst_out = inst_out[:real_num_f, :real_num_f] # padding faces are discarded

        bottom_out = np.array(r_Bottom_Face_Recogintion, dtype=np.float)
        bottom_out = bottom_out[:real_num_f] # padding faces are discarded
        # print(seg_out.shape, inst_out.shape, bottom_out.shape)
        # features = post_process((seg_out, inst_out, bottom_out), inst_thres=INST_THRES, bottom_thres=BOTTOM_THRES)
        emb_out = embedding_layers_output[0][:real_num_f, :]
        dedu_out = deduction_layers_output[0][:real_num_f, :]
        features = post_process2(inst_out, seg_out, bottom_out, real_num_f, emb_out, dedu_out)
        # calculate recognition performance
        # print(test_similar_matrix[i, :].shape, test_seg_labels[i, :].shape, test_bottom_labels[i, :].shape)
        sim_mat = test_similar_matrix[i, :]
        sim_mat = sim_mat[:real_num_f, :real_num_f] 
        seg_list = np.argmax(test_seg_labels[i, :], axis=1)
        seg_list = seg_list[:real_num_f] 
        bottom_seg_list = test_bottom_labels[i, :]
        bottom_seg_list = bottom_seg_list[:real_num_f] 
        # print(sim_mat.shape, seg_list.shape, bottom_seg_list.shape)
        labels = parser_label(sim_mat, seg_list, bottom_seg_list)

        pred, gt, tp = cal_recognition_performance(features, labels)
        rec_predictions += pred
        rec_truelabels += gt
        rec_truepositives += tp

        # calculate localization performance
        pred, gt, tp = cal_localization_performance(features, labels)
        loc_predictions += pred
        loc_truelabels += gt
        loc_truepositives += tp

    mean_test_seg_acc = test_seg_acc.compute().item()
    mean_test_seg_iou = test_seg_iou.compute().item()
    mean_test_inst_acc = test_inst_acc.compute().item()
    mean_test_inst_f1 = test_inst_f1.compute().item()
    mean_test_bottom_acc = test_bottom_acc.compute().item()
    mean_test_bottom_iou = test_bottom_iou.compute().item()
    
    print(f'test_seg_acc: {mean_test_seg_acc*100}, \
            test_seg_iou: {mean_test_seg_iou*100}, \
            test_inst_acc: {mean_test_inst_acc*100}, \
            test_inst_f1: {mean_test_inst_f1*100}, \
            test_bottom_acc: {mean_test_bottom_acc*100}, \
            test_bottom_iou: {mean_test_bottom_iou*100}')
    
    print('------------- recognition performance------------- ')
    print('rec_pred', rec_predictions)
    print('rec_true', rec_truelabels)
    print('rec_trpo', rec_truepositives)
    precision, recall = eval_metric(rec_predictions, rec_truelabels, rec_truepositives)
    print('recognition Precision scores')
    # print precision for each class
    print_class_metric(precision)
    precision = precision.mean()
    print('AVG recognition Precision:', precision)
    print('recognition Recall scores')
    # print recall for each class
    print_class_metric(recall)
    recall = recall.mean()
    print('AVG recognition Precision:', recall)
    print('recognition F scores')
    rec_F = (2*recall*precision)/(recall+precision)
    print(rec_F)

    print('------------- localization performance------------- ')
    print('loc_pred', loc_predictions)
    print('loc_true', loc_truelabels)
    print('loc_trpo', loc_truepositives)
    precision, recall = eval_metric(loc_predictions, loc_truelabels, loc_truepositives)
    print('localization Precision scores')
    # print precision for each class
    print_class_metric(precision)
    precision = precision.mean()
    print('AVG localization Precision:', precision)
    print('localization Recall scores')
    # print recall for each class
    print_class_metric(recall)
    recall = recall.mean()
    print('AVG localization Precision:', recall)
    print('localization F scores')
    loc_F = (2*recall*precision)/(recall+precision)
    print(loc_F)


    print('------------- Final ------------- ')
    print('rec F scores(%):', rec_F*100)
    print('loc F scores(%):', loc_F*100)


