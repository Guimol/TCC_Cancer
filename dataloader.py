import numpy as np
import os
import tensorflow as tf

import random
from glob import glob
# from tqdm import tqdm
import tifffile
import cv2
import matplotlib.pyplot as plt

import typing

import albumentations

def img_is_color(img):

  if len(img.shape) == 3:
      # Check the color channels to see if they're all the same.
      c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
      if (c1 == c2).all() and (c2 == c3).all():
          return True

  return False

img_types = [".tif", ".tiff"]
BENIGN = 0
MALIGNANT = 1

class CustomDataGenerator(tf.keras.utils.Sequence):
  
  def __init__(self, 
               data: list,
               batch_size: int = 32,
               input_size: tuple = (224, 224, 3),
               shuffle: bool = True,
               normalize: bool = False,
               transform_imgs: bool = True):
      
    self.batch_size = batch_size
    self.input_size = input_size
    self.shuffle = shuffle
    self.normalize = normalize
    self.img_paths = data
    self.transform_imgs = transform_imgs
      
  def on_epoch_end(self):
    if self.shuffle:
      random.shuffle(self.img_paths)
      
  def get_images(self, count: int, transform_imgs: bool=True) -> typing.Tuple[typing.List[np.array], typing.List[int], typing.List[dict]]:
    
    default_transform_imgs = self.transform_imgs
    self.transform_imgs = transform_imgs
    
    batches = self.img_paths[0:count]
    X, y, replay_transforms = self.__get_data(batches)
    
    # Return transforms to default value
    self.transform_imgs = default_transform_imgs
    
    return X, y, replay_transforms
  
  def __show_transforms(self, replay: dict) -> str:
    
    transforms = replay["transforms"]
    applied_transforms = [transform for transform in transforms if transform["applied"] == True]
    
    names = [transform["__class_fullname__"].split(".")[-1] for transform in applied_transforms]
    
    str_transforms = " | ".join(names)
    
    return "\n" + str_transforms
  
  def visualize_samples(self, 
                        count: int=9, 
                        transform_imgs: bool=True, 
                        list_titles: typing.List[str]=None,
                        list_cmaps: typing.List[np.array]=None,
                        grid: bool=True, 
                        num_cols: int=3, 
                        figsize: typing.Tuple[int, int]=(20, 10), 
                        title_fontsize: int=20, 
                        mode: str="write",
                        save_path: str=None):
    '''
    Shows a grid of images, where each image is a Numpy array. The images can be either
    RGB or grayscale.

    Parameters:
    ----------
    count: int
        Number of the images to be displayed.
    list_titles: list or None
        Optional list of titles to be shown for each image.
    list_cmaps: list or None
        Optional list of cmap values for each image. If None, then cmap will be
        automatically inferred.
    grid: boolean
        If True, show a grid over each image.
    num_cols: int
        Number of columns to show.
    figsize: tuple of width, height
        Value to be passed to pyplot.figure()
    title_fontsize: int
        Value to be passed to set_title().
    mode: "write" or "show"
        Chooses whether to save plot in disk or show in terminal.
    save_path: str or None
        If mode == "write" specify the path to save the plot, if None get current working dir
    '''
    
    images_array, classes_array, replay_array = self.get_images(count, transform_imgs)
    
    list_images = [img for img in images_array]
    list_classes = [img_class for img_class in classes_array]
    list_replay_tranforms = [replay for replay in replay_array]

    assert isinstance(list_images, list)
    assert len(list_images) > 0
    assert isinstance(list_images[0], np.ndarray)

    if list_titles is not None:
        assert isinstance(list_titles, list)
        assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))
    
    if list_cmaps is not None:
        assert isinstance(list_cmaps, list)
        assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))
    
    num_images  = len(list_images)
    num_cols    = min(num_images, num_cols)
    num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

    # Create a grid of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Create list of axes for easy iteration.
    if isinstance(axes, np.ndarray):
        list_axes = list(axes.flat)
    else:
        list_axes = [axes]

    for i in range(num_images):

        img    = list_images[i]
        if list_titles is not None:
          title = list_titles[i]
        else:
          applied_transforms = ""
          if list_replay_tranforms[i] is not None:
            applied_transforms = self.__show_transforms(list_replay_tranforms[i])
          title = f"Image {i} | Class {list_classes[i]}" + applied_transforms
        cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
        list_axes[i].imshow(img, cmap=cmap)
        list_axes[i].set_title(title, fontsize=title_fontsize) 
        list_axes[i].grid(grid)

    for i in range(num_images, len(list_axes)):
        list_axes[i].set_visible(False)
        
    fig.tight_layout()
    
    if mode == "show":
      _ = plt.show()
    elif mode == "write":
      if save_path == None:
        save_path = os.path.join(os.getcwd(), f"visualize_plot_{'with' if transform_imgs else 'no'}_transforms.png")
      
      try:
        plt.savefig(save_path)
      except Exception as error:
        print(f"Error in saving the plot: {error}")
    
  def __getitem__(self, index: int) -> tuple:
    
    batches = self.img_paths[index * self.batch_size: (index + 1) * self.batch_size]
    X, y, _ = self.__get_data(batches)
    
    return X, y
    
  def __get_data(self, batches: list) -> typing.Tuple[np.array, np.array]:
    
    # Sample (tuple) -> (img_path, [list of transformations])
    
    data_batch = [self.__get_input(sample) for sample in batches]
    
    img_batch = np.asarray([data[0] for data in data_batch])
    replay_batch = [data[1] for data in data_batch]
    
    #! Temp workaround
    class_batch = np.asarray([get_output(sample[0] if isinstance(sample, tuple) else sample) for sample in batches])
    
    return img_batch, class_batch, replay_batch
  
  def __get_input(self, sample: tuple) -> np.array:
    
    #! Temp workaround
    if isinstance(sample, tuple):
      path = sample[0]
    elif isinstance(sample, str):
      path = sample
    
    replay = None
    
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
    if self.transform_imgs:
      data = sample[1](image=img_array)
      img_array = data["image"]
      replay = data["replay"]
    
    if self.normalize: 
      img_array = img_array / 255.0
    
    return img_array, replay
  
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
  
  transforms = albumentations.ReplayCompose([])
  
  for choice in random.sample(transformations_list, number_transforms):
    transforms = albumentations.ReplayCompose([*transforms, choice])
  
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

def create_folds(pacients: dict, number_folds: int) -> list:
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

def separate_test_pacients(pacients: dict, split_percentage: int) -> typing.Tuple[dict, list]:
  pacients_amount = sum([len(pacients[type]) for type in pacients])
  
  test_pacients_amount = int(pacients_amount * (split_percentage/100))
  
  pacients_per_type_amount = max(int(test_pacients_amount / len(pacients)), 1)
  
  test_pacients = []
  copy_pacients = pacients.copy()
  
  for type in pacients:
    for _ in range(pacients_per_type_amount):
      key = list(copy_pacients[type].keys())[0]
      if key == "count": 
        key = list(copy_pacients[type].keys())[1]
      test_pacients += extract_pacient_paths(copy_pacients[type].pop(key))
  
  return copy_pacients, test_pacients

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
  albumentations.ColorJitter(brightness=(0.2), contrast=(0.3), saturation=(0.3), hue=(-0.1, 0.1), p=params["transformations"]["ColorJitter"]),
  albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=params["transformations"]["GaussianBlur"]),
  albumentations.ShiftScaleRotate(shift_limit=(0.05, 0.2), scale_limit=(-0.1, 0.5), rotate_limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_WRAP, p=params["transformations"]["ShiftScaleRotate"]),
  albumentations.RandomSnow(p=params["transformations"]["RandomSnow"])
]

img_paths = scan_datasets(datasets_paths)

pacients = retrieve_pacients(img_paths)

train_pacients, test_pacients = separate_test_pacients(pacients, 5)

folds = create_folds(train_pacients, 5)
upsampled_folds = upsample_folds(folds, upsampling_factor=5, transformations_list=transformations_list)

# Use case for Train, Validation and Test
#! Not working
train, test = split_folds_train_test(upsampled_folds, 90, 0)
train, val = split_folds_train_test(train, 78, 0)

import pdb; pdb.set_trace()

# Use case for Cross Validation
for index, (train, test) in enumerate(PermuteFolds(upsampled_folds)):
    
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