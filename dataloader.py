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
    
    class_batch = np.asarray([self.__get_output(sample[0]) for sample in batches])
    
    return img_batch, class_batch, replay_batch
  
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
    
    if self.normalize: img_array = img_array / 255.0
    
    return img_array, replay
  
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
  
  transforms = albumentations.ReplayCompose([])
  
  for choice in random.sample(transformations_list, number_transforms):
    transforms = albumentations.ReplayCompose([*transforms, choice])
  
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
    
  train_keys = pacients_keys[0:int((split_percentage/100) * len(pacients_keys))]
  test_keys = pacients_keys[int((split_percentage/100) * len(pacients_keys)):]
      
  return train_keys, test_keys

'''
# Usage
datasets_paths = ["/home/guilherme/Downloads/breast_20x", "/home/guilherme/Downloads/mouth_20x"]

params = {
  "batch_size": 16,
  "epochs": 200,
  "input_size": (224, 224, 3),
  "learning_rate": 1e-3,
  "transformations": {"ColorJitter": 0.85,
                      "GaussianBlur": 0.85,
                      "ShiftScaleRotate": 1,
                      "RandomSnow": 0.3}
}

transformations_list = [
  albumentations.ColorJitter(brightness=(0.2), contrast=(0.3), saturation=(0.3), hue=(-0.1, 0.1), p=params["transformations"]["ColorJitter"]),
  albumentations.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=params["transformations"]["GaussianBlur"]),
  albumentations.ShiftScaleRotate(shift_limit=(0.05, 0.2), scale_limit=(-0.1, 0.5), rotate_limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_WRAP, p=params["transformations"]["ShiftScaleRotate"]),
  albumentations.RandomSnow(p=params["transformations"]["RandomSnow"])
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