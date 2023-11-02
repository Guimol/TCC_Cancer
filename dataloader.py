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
    
    class_batch = np.asarray([self.__get_output(sample[0]) for sample in batches])
    
    return img_batch, class_batch
  
  def __get_output(self, path: str) -> int:
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
    
    # Sample[1] = transformations to be applied in image
    transformed_img = sample[1](image=img_array)["image"]
    
    if self.normalize: transformed_img = transformed_img / 255.0
    
    return transformed_img
  
  def __len__(self):
      return len(self.img_paths) // self.batch_size
    
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
  
  for _ in range(number_transforms):
    transforms = albumentations.Compose([*transforms, random.choice(transformations_list)])
  
  return transforms
    
def scan_datasets(datasets_paths: list, upsampling_factor: int=1, transformations_list: list=[None], random_seed: int=None, min_transformations: int=1) -> dict:
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
    
    img_paths_and_transformations = {}
    
    if upsampling_factor > 0:
      for i in range(upsampling_factor):
        for index, img_path in enumerate(img_paths):
          img_paths_and_transformations.update({f"{index}_{i}": {"path": img_path,
                                                                 "transformations": select_transformations(transformations_list, random_seed, min_transformations)}})
    
    # New line
    print()
  
  return img_paths_and_transformations

def retrieve_pacients(data: dict) -> dict:
  pacients = {}
  
  for id in data:
      path = data[id]["path"]
      
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
              try:
                pacients[f"breast_{filename}"].append(id)
              except:
                pacients.update({f"breast_{filename}": [id]}) # id
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
          try:
              pacients[f"mouth_{mouth_id}"].append(id)
          except:
            pacients.update({f"mouth_{mouth_id}": [id]})
        else:
          continue
  
  return pacients

def retrieve_pacients_data(pacients: dict, data: dict, keys: typing.List[int]) -> typing.List[typing.Tuple[str, list]]:
  retrieved_data = []
  
  for key in keys:
    if type(pacients[key]) == list:
      for pacient_id in pacients[key]:
        retrieved_data.append((data[pacient_id]["path"], data[pacient_id]["transformations"]))
    else:
      retrieved_data.append((data[pacients[key]]["path"], data[pacients[key]]["transformations"]))
  
  return retrieved_data

def split_pacients_train_test(pacients: typing.Union[dict, list], split_percentage: int) -> typing.Tuple[typing.List[int], typing.List[int]]:
  
  if type(pacients) == dict:
    pacients_keys = list(pacients.keys())
  elif type(pacients) == list:
    pacients_keys = pacients
  else:
    print("Invalid type for pacients")
    quit()
  
  random.shuffle(pacients_keys)
  
  train_keys = pacients_keys[0:int((split_percentage/100) * len(pacients_keys))]
  test_keys = pacients_keys[int((split_percentage/100) * len(pacients_keys)):]
      
  return train_keys, test_keys

'''
# Usage
datasets_paths = ["/home/guilherme/Downloads/breast_20x", "/home/guilherme/Downloads/mouth_20x"]

transformations_list = [
  albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
  albumentations.ColorJitter(brightness=(0.1, 0.5), contrast=(0.1, 0.5), saturation=(0.1, 0.5), hue=(-0.2, 0.2), p=0.5),
  albumentations.ShiftScaleRotate(shift_limit=(0.05, 0.2), scale_limit=(-0.1, 0.5), rotate_limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_WRAP, p=0.5)
]

img_paths = scan_datasets(datasets_paths, 5, transformations_list)

pacients = retrieve_pacients(img_paths)

train_keys, test_keys = split_pacients_train_test(pacients, 90)
train_keys, val_keys = split_pacients_train_test(train_keys, 80)

train = retrieve_pacients_data(pacients, img_paths, train_keys)
val = retrieve_pacients_data(pacients, img_paths, val_keys)
test = retrieve_pacients_data(pacients, img_paths, test_keys)
    
cdg_train = CustomDataGenerator(
  data=train,
  batch_size=1,
  input_size=(480, 480, 3))

# cdg_val = CustomDataGenerator(
#   data=val,
#   batch_size=16)

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