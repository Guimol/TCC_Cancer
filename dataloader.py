import numpy as np
import os
import tensorflow as tf

import random
from glob import glob
# from tqdm import tqdm
import tifffile
import cv2

import typing

import albumentations

img_types = [".tif", ".tiff"]
BENIGN = 0
MALIGNANT = 1

class CustomDataGenerator(tf.keras.utils.Sequence):
  
  def __init__(self, 
               data: list,
               batch_size: int = 32,
               input_size: tuple = (224, 224, 3),
               shuffle: bool = True,
               normalize: bool = False):
      
    self.batch_size = batch_size
    self.input_size = input_size
    self.shuffle = shuffle
    self.normalize = normalize
    self.img_paths = data
      
  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.img_paths)
    
  def __getitem__(self, index: int) -> tuple:
    
    batches = self.img_paths[index * self.batch_size: (index + 1) * self.batch_size]
    X, y = self.__get_data(batches)
    
    return X, y
    
    
  def __get_data(self, batches: list) -> tuple:
    
    img_batch = np.asarray([self.__get_input(sample) for sample in batches])
    
    class_batch = np.asarray([get_output(sample[0]) for sample in batches])
    
    return img_batch, class_batch
  
  def __get_input(self, sample: tuple) -> np.array:
    path = sample[0]
    
    img_format = "." + path.split(".")[-1]
    
    if img_format in [".tif", ".tiff"]:
      img = tifffile.imread(path)
    else:
      img = cv2.imread(path)
      
    # Deprecated
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = tf.image.resize(img, (self.input_size[0], self.input_size[1])).numpy()
    img_array = cv2.resize(img, (self.input_size[0], self.input_size[1]))
    
    # sample[1] are the transformations to be applied
    transformed_img = sample[1](image=img_array)["image"]
    
    if self.normalize: transformed_img = transformed_img / 255.0
    
    return transformed_img
  
  def __len__(self):
      return len(self.img_paths) // self.batch_size

def get_output(path: str) -> int:
    # 0 is benign
    # 1 is malignant
    
    filename = path.split("/")[-1].split(".")[0]

    # Hardcoded for the folders names
    if "breast" in path:
      
      # Breast files labeled as <number>_<odd> malignant and <number>_<even> benign
      filename_split = filename.split("_")
      
      if int(filename_split[1]) % 2 != 0:
        return np.asarray(1)
      else:
        return np.asarray(0)
      
    elif "mouth" in path:
      # Mouth files labeled as ImageL<number>C<number>_<ID>_<0(benign)|1(malignant)>
      filename_split = filename.split("_")
      
      return np.asarray(int(filename_split[2]))

    
def select_transformations(transformations_list: list, random_seed: int=None, min_transformations: int=1) -> list:
  '''
  Given a list of transformations select at least N of them to be applied to an image
  Inputs: transformations_list: list
          random_seed: int
          min_transformations: int
  Output: transforms: albumentations.Compose()
  '''
  
  random.seed(random_seed)
  
  number_transforms = random.randint(min_transformations, len(transformations_list) - 1)
  
  transforms = albumentations.Compose([])
  
  for choice in random.sample(transformations_list, number_transforms):
    transforms = albumentations.Compose([*transforms, choice])
  
  return transforms
    
def scan_datasets(datasets_paths: list) -> dict:
  img_paths = []
  
  for index, path in enumerate(datasets_paths):
    print(f"[{index + 1}/{len(datasets_paths)}] Scanning images folder \"{path}\": ", end="")
    
    # Store the amount of previous iteration
    prev_img_paths_len = 0
    curr_dir_img_paths = []
    
    # Scan the given paths for the valid img types
    for img_type in img_types:
      curr_dir_img_paths.extend(glob(os.path.join(path, "*" + img_type)))
      if len(curr_dir_img_paths) - prev_img_paths_len > 0:
        print(f"Found {len(curr_dir_img_paths) - prev_img_paths_len} {img_type} images; ", end="")
      prev_img_paths_len = len(curr_dir_img_paths)
    
    # Append the newly found paths to class img_paths list
    img_paths.extend(curr_dir_img_paths)
    
    # New line
    print()
  
  return img_paths

def upsample_folds(folds: typing.List[list], upsampling_factor: int=1, transformations_list: list=[None], random_seed: int=None, min_transformations: int=1) -> typing.List[list]:
  upsampled_folds = []
  
  for _ in range(len(folds)):
    upsampled_folds.append([])
  
  if upsampling_factor > 0:
      for index, fold in enumerate(folds):
        for _ in range(upsampling_factor):
          for img_path in fold:
            upsampled_folds[index].append((img_path, 
                                          select_transformations(transformations_list, random_seed, min_transformations)))
  else:
    print(f"Upsample factor \"{upsampling_factor}\" not permitted.")
    quit()
  
  return upsampled_folds

def retrieve_pacients(data: dict) -> dict:
  pacients = {"mouth": {"count": {BENIGN: 0, MALIGNANT: 0}}, "breast": {"count": {BENIGN: 0, MALIGNANT: 0}}}
    
  for path in data:
  # for id in data
      # path = data[id]["path"]
      
      # Breast <Num>_<odd/even>
      # Pacient = <Num>_<odd> + <Num>_<even>
      if "breast" in path:
        filename = path.split("/")[-1].split(".")[0]
        filename_split = filename.split("_")
        
        if int(filename_split[1]) % 2 != 0:
          for index in range(int(filename_split[1]), int(filename_split[1]) + 2):
            new_name = "".join([filename_split[0], "_", str(index)])
            new_path = path.replace(filename, new_name)
            if os.path.isfile(new_path):
              pacient_class = int(get_output(new_path))
              try:
                pacients["breast"][filename][pacient_class].append(new_path) # id
              except:
                pacients["breast"].update({filename: {BENIGN: [], MALIGNANT: []}}) # id
                pacients["breast"][filename][pacient_class].append(new_path)
              
              # Increment count for pacient class
              pacients["breast"]["count"][pacient_class] += 1
            else:
              continue
        else:
          continue

      # Mouth ImageLXCY_<ID>_<0/1>
      # Pacient = ImageLXCY_<ID>_<0/1> + ... + ImageLXCY_<ID + 4>_<0/1>
      if "mouth" in path:
        filename = path.split("/")[-1].split(".")[0]
        filename_split = filename.split("_")
        mouth_id = filename_split[1]
        if os.path.isfile(path):
          pacient_class = int(get_output(path))
          try:
              pacients["mouth"][mouth_id][pacient_class].append(path) # id 
          except:
            pacients["mouth"].update({mouth_id: {BENIGN: [], MALIGNANT: []}}) # id
            pacients["mouth"][mouth_id][pacient_class].append(path)
            
          # Increment count for pacient class
          pacients["mouth"]["count"][pacient_class] += 1
        else:
          continue
  
  return pacients

def extract_pacient_paths(pacient: dict) -> list:
  paths = []
  
  for type in [BENIGN, MALIGNANT]:
    if isinstance(pacient[type], list):
      for path in pacient[type]:
        paths.append(path)
    else:
      paths.append(pacient[type])
      
  return paths

def min_list_length_index(list_of_lists):
    min_length = float('inf')  # Initialize with positive infinity
    min_index = None

    for i, sublist in enumerate(list_of_lists):
        current_length = len(sublist)
        if current_length < min_length:
            min_length = current_length
            min_index = i

    return min_index

def create_folds(pacients: dict, data: dict, number_folds: int) -> list:
  folds = []
  
  for _ in range(number_folds):
    folds.append([])
    
  copy_pacients = pacients.copy()
  
  for type in pacients:
    #! Not used
    # benign_percent = pacients[type]["count"][BENIGN] / sum(pacients[type]["count"].values())
    # malignant_percent = 1 - benign_percent
    
    pacients_per_fold = (len(pacients[type]) - 1) // number_folds
    leftover_pacients = (len(pacients[type]) - 1) % number_folds
    
    #! Not used
    # benign_pacients = int(pacients_per_fold * benign_percent)
    # malignant_pacients = pacients_per_fold - benign_pacients
    
    all_pacients_keys = list(pacients[type].keys())
    
    for index, fold in enumerate(folds):
      pacients_keys = all_pacients_keys[index * pacients_per_fold : (index + 1) * pacients_per_fold]
            
      try:
        pacients_keys.remove("count")
        if leftover_pacients > 0:
          pacients_keys.append(all_pacients_keys.pop())
      except:
        pass
      
      for key in pacients_keys:
        fold += extract_pacient_paths(copy_pacients[type].pop(key))
    
    if len(copy_pacients[type]) > 1:
      
      leftover_pacients_keys = list(copy_pacients[type].keys())
      leftover_pacients_keys.remove("count")
      
      for key in leftover_pacients_keys:
      
        fold_index = min_list_length_index(folds)
        folds[fold_index] += extract_pacient_paths(copy_pacients[type].pop(key))
  
  return folds

def split_folds_train_test(folds: typing.List[list], split_percentage: int, random_seed: int=None) -> typing.Tuple[list, list]:
  #! Refactor to use pacients instead of folds
  
  data = []
  
  if isinstance(folds[0], list):
    for fold in folds:
      data += fold
  elif isinstance(folds[0], tuple):
    data = folds
  else:
    print("Invalid type for fold")
    quit()
    
  random.seed(random_seed)
  random.shuffle(data)
  
  train = data[0:int((split_percentage/100) * len(data))]
  test = data[int((split_percentage/100) * len(data)):]
      
  return train, test

class PermuteFolds:
  def __init__(self, folds):
    self.folds = folds
    self.indexes = list(range(len(folds)))
    self.current_index = -1
  
  def __iter__(self):
    return self
  
  def __next__(self) -> typing.Tuple[list, list]:
    self.current_index += 1
    
    if self.current_index < len(self.folds):
    
      test = self.folds[self.current_index]
      
      train = []
      train_folds = [fold for index, fold in enumerate(self.folds) if index != self.current_index]
      
      for fold in train_folds:
        train += fold
            
      return train, test
      
    else:
      raise StopIteration

'''
# Usage
datasets_paths = ["/home/guilherme/Downloads/breast_20x", "/home/guilherme/Downloads/mouth_20x"]

# When combining the datasets we get the following distribution
# 0 is benign
# 1 is malignant

#Malignant: 76.78% (172/224)
#Benign: 23.21% (52/224)

transformations_list = [
  albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
  albumentations.ColorJitter(brightness=(0.1, 0.5), contrast=(0.1, 0.5), saturation=(0.1, 0.5), hue=(-0.2, 0.2), p=0.5),
  albumentations.ShiftScaleRotate(shift_limit=(0.05, 0.2), scale_limit=(-0.1, 0.5), rotate_limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_WRAP, p=0.5)
]

img_paths = scan_datasets(datasets_paths)

pacients = retrieve_pacients(img_paths)
#! Separate pacients for testing

folds = create_folds(pacients, img_paths, 5)
upsampled_folds = upsample_folds(folds, upsampling_factor=5, transformations_list=transformations_list)

# Use case for Train, Validation and Test

train, test = split_folds_train_test(upsampled_folds, 90, 0)
train, val = split_folds_train_test(train, 78, 0)

# Use case for Cross Validation
for train, test in PermuteFolds(upsampled_folds):
    
  cdg_train = CustomDataGenerator(
    data=train,
    batch_size=16,
    input_size=(480, 480, 3))

  cdg_test = CustomDataGenerator(
    data=test,
    batch_size=16,
    input_size=(480, 480, 3))

# Reference for dataloader functions from TF

# import pathlib
# dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
# data_dir = pathlib.Path(archive).with_suffix('')

# batch_size = 32
# img_height = 224
# img_width = 224

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=42,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)

# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=42,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
'''
