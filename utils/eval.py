import numpy as np
from parseDataset import *

class EvaluatePasadenaUrbanTrees:
  def __init__(self, dataset, distance_threshold, dataset_id="test"):
    """
    Constructor of helper class for evaluating detection performance
    :param dataset (ParseDataset): Loaded dataset dataset=ParseDataset(dir_name)
    :param distance_threshold (float): maximum distance in meters to match a
       detected tree to a ground truth one
    :param dataset_id (str): dataset name (e.g., "test","val")
    :return:
    """
    self.dataset = dataset
    self.distance_threshold = distance_threshold
    self.dataset_id = dataset_id
  
  def evaluate_detection_average_precision(self, scored_detections):
    """
    Evaluate detection performance in terms of average precision.  We will
    Enumerate all detections in order of detection confidence, and compute
    the precision for all values of recall.  Trees marked
    as difficult are not penalized for false positives or negatives.
    :param scored_detections (list): a list of detections.  Each detection
       should have fields 'lat', 'lng', 'score' which define its geolocation
       and detection confidence score.  
    :return: 
      average_precision (float): the average precision
      precisions (list): an array of precision values
      recalls (list) an array of recall values for each value in precisions
      matches (list): a list of 2-tuples, where the first entry of each tuple
        is an index into scored_detections, and the 2nd entry is an index into
        the ground truth datast (self.detection_datasets[dataset_id])
    """
    # Separate the ground truth trees into difficult and non-difficult trees
    gt_dataset = self.dataset.detection_datasets[self.dataset_id]
    gtcoords, gtinds, gtcoords_diff, gtinds_diff = [], [], [], []
    for i in range(len(gt_dataset)):
      if gt_dataset[i]['diff']: 
        gtcoords_diff.append((gt_dataset[i]['lat'], gt_dataset[i]['lng']))
        gtinds_diff.append(i)
      else:
        gtcoords.append((gt_dataset[i]['lat'], gt_dataset[i]['lng']))
        gtinds.append(i)

    num_gt = len(gtinds)
    precisions, recalls, matches, num, true_positives = [], [], [], 0.0, 0.0
    det_inds = sorted(enumerate(scored_detections), 
                      key=lambda d: -d[1]['score'])
    ranked_detections = [scored_detections[i[0]] for i in det_inds]
    threshold = ranked_detections[-1]['score']
    for i in range(len(ranked_detections)):
      # Try to match the next highest scoring to detection to a good tree
      # within the required distance_threshold, if one exists
      p = ranked_detections[i]
      if len(gtcoords) > 0:
        dists = array_haversine_distances(p['lat'], p['lng'], 
                            np.asarray(gtcoords), max_dist=self.distance_threshold)
        j = dists.argmin()
      else:
        j = -1
      if j >= 0 and dists[j] < self.distance_threshold:
        matches.append((det_inds[i][0], gtinds[j]))
        true_positives += 1.0
        num += 1.0
        precisions.append(true_positives / num)
        recalls.append(true_positives / float(num_gt))

        # Remove matched ground truth object, so it can't be matched multiple 
        # times
        gtcoords[j] = gtcoords[-1]
        gtinds[j] = gtinds[-1]
        gtcoords.pop()
        gtinds.pop()
      else:
        # If we couldn't match the detection to a good tree, try to match it
        # to a difficult tree
        if len(gtcoords_diff)>0:
          dists_diff = array_haversine_distances(p['lat'], p['lng'], 
                         np.asarray(gtcoords_diff), max_dist=self.distance_threshold)
          j = dists_diff.argmin()
        else:
          j = -1
        if j >= 0 and dists_diff[j] < self.distance_threshold:
          # If this detection matches to a difficult example, don't count it 
          # as a false positive
          matches.append((det_inds[i], gtinds_diff[j]))
          
          # Remove matched ground truth difficult object, so it can't be 
          # matched multiple times
          gtcoords_diff[j] = gtcoords_diff[-1]
          gtinds_diff[j] = gtinds_diff[-1]
          gtcoords_diff.pop()
          gtinds_diff.pop()
        else:
          num += 1.0
    
    # Dividing by the number of ground truth examples (instead of number
    # of elements in precisions) means we say we have 0 precision for 
    # unrecalled ground truth objects
    ave_precision = np.asarray(precisions).sum() / float(num_gt)
    return ave_precision, precisions, recalls, matches
  

  def evaluate_classification(self, preds, useClassSubset=True):
    """
    Evaluate classification performance in terms of average class precision,
    average class recall, and total dataset-level precision
    :param preds (list): a list of class ids.  Each  should correspond to
      the class prediction of an example in 
      self.classification_datasets[dataset_id]
    :return:
      average_precision (float): the class average precision
      average_recall (float): the class average recall
      raw_precision (float): the dataset average precision
      precisions (list): an array of precision values for each class
      recalls (list) an array of recall values for each value in precisions
      class_labels (list): an array of class labels for each value in precisions
      confMat (np.array): the class confusion matrix
    """
    if useClassSubset:
      species_subset = self.dataset.species_subset
    else:
      species_subset = list(range(len(self.dataset.tree_species)))
    gt_dataset = self.dataset.classification_datasets[self.dataset_id]
    num_classes = len(species_subset)
    class_inds, class_labels = {}, []
    for c in species_subset: 
      class_inds[c] = len(class_ids)
      class_labels.append(self.dataset.tree_species[c])
    confMat = np.zeros((num_classes,num_classes))
    for i in range(len(gt_dataset)):
      if not gt_dataset[i]['diff']:
        confMat[class_inds[gt_dataset[i]['class_id']],class_inds[preds[i]]] += 1
    valid = np.where(confMat.sum(axis=1)>0)[0] # classes with at least 1 test ex
    precisions = np.diag(confMat) / np.maximum(confMat.sum(axis=0),.0001)
    recalls = np.diag(confMat) / np.maximum(confMat.sum(axis=1),.0001)
    ave_precision = precisions[valid].mean()
    ave_recall = recalls[valid].mean()
    raw_precision = np.diag(confMat).sum() / confMat.sum()
    return ave_precision, ave_recall, raw_precision, precisions, recalls, class_labels, confMat

