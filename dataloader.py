import numpy as np
import os
import tensorflow as tf

import random
from glob import glob
# from tqdm import tqdm
import tifffile
import cv2

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
    
    img_batch = np.asarray([self.__get_input(path) for path in batches])
    
    class_batch = np.asarray([self.__get_output(path) for path in batches])
    
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
   
  def __get_input(self, path: str) -> np.array:
    img_format = "." + path.split(".")[-1]
    
    if img_format in [".tif", ".tiff"]:
      img = tifffile.imread(path)
    else:
      img = cv2.imread(path)
      
    # Deprecated
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    # img_array = tf.image.resize(img, (self.input_size[0], self.input_size[1])).numpy()
    img_array = cv2.resize(img, (self.input_size[0], self.input_size[1]))
    
    if self.normalize: img_array = img_array / 255.0
    
    return img_array
  
  def __len__(self):
      return len(self.img_paths) // self.batch_size
    
def scan_datasets(datasets_paths: list) -> list:
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

def split_train_val(img_paths: list, split_percentage: int):
    pacients = {}

    for path in img_paths:
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
                pacients[filename].append(new_path)
              except:
                pacients.update({filename: [new_path]})
            else:
              continue
        else: 
          continue

      # Mout ImageLXCY_<ID>_<0/1>
      # Pacient = ImageLXCY_<ID>_<0/1> + ... + ImageLXCY_<ID + 4>_<0/1>
      if "mouth" in path:
        filename = path.split("/")[-1].split(".")[0]
        filename_split = filename.split("_")
        mouth_id = filename_split[1]
        if os.path.isfile(path):
          try:
            if "breast" in pacients[mouth_id][0]:
              pacients.update({len(pacients) + 1: [path]})
            else:
              pacients[mouth_id].append(path)
          except:
            pacients.update({mouth_id: [path]})
        else:
          continue
        
    pacients_keys = list(pacients.keys())
    random.shuffle(pacients_keys)
    
    train_keys = pacients_keys[0:int((split_percentage/100) * len(pacients_keys))]
    val_keys = pacients_keys[int((split_percentage/100) * len(pacients_keys)):]
    
    train = []
    val = []
    
    for key in train_keys:
      if type(pacients[key]) == list:
        for value in pacients[key]:
          train.append(value)
      else:
        train.append(pacients[key])
        
    #! Duplicate
    for key in val_keys:
      if type(pacients[key]) == list:
        for value in pacients[key]:
          val.append(value)
      else:
        val.append(pacients[key])
        
    return train, val

''' # Usage
datasets_paths = ["/home/guilherme/Downloads/breast_20x", "/home/guilherme/Downloads/mouth_20x"]

img_paths = scan_datasets(datasets_paths)
train, val = split_train_val(img_paths, 60)
val, test = split_train_val(val, 75)
    
cdg_train = CustomDataGenerator(
  data=train,
  batch_size=16)

cdg_val = CustomDataGenerator(
  data=val,
  batch_size=16)

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