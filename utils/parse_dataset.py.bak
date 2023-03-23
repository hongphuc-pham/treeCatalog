import os
import json
from image_extract import *
from eval import *

class ParseDataset:
  def __init__(self, dataset_dir):
    """
    Constructor of Pasadena Urban Trees helper class for loading the dataset, extracting images from geographical
    locations, 
    :param dataset_dir (str): location of the PasadenaUrbanTrees folder
    :return:
    """
    self.dataset_dir = dataset_dir

    self.parse_dataset()
    self.aerial_extractor = AerialExtractor(self.dataset_dir)
    self.streetview_extractor = StreetViewExtractor(self.dataset_dir, self.panos)
    self.evaluator = EvaluatePasadenaUrbanTrees(self, 4)

  def parse_dataset(self):
    """
    Parse the entire dataset
    """
    with open(os.path.join(self.dataset_dir, "detection_datasets.json")) as f:
      self.detection_datasets = json.load(f)
    with open(os.path.join(self.dataset_dir, "classification_datasets.json")) as f:
      self.classification_datasets = json.load(f)
    with open(os.path.join(self.dataset_dir, "panos.json")) as f:
      self.panos = json.load(f)
    with open(os.path.join(self.dataset_dir, "tree_species.json")) as f:
      self.tree_species = json.load(f)
    with open(os.path.join(self.dataset_dir, "rois.json")) as f:
      self.rois = json.load(f)
