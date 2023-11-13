import os
import comet_ml

print("Imported comet_ml")

model_name = "resnet" # models: ["resnet", "efficientnet", "densenet", "transformer"]
training_mode = "transfer_learning" # modes: ["transfer_learning", "fine_tuning", "from_scratch"]
weight_path = "/home/guilherme/tcc_guilherme/densenet_no_preprocessing_transfer_learning_v1/checkpoints/ckpt_" # only used if training_mode == "fine_tuning"
saved_model_path = "/home/guilherme/tcc_guilherme/densenet_no_preprocessing_transfer_learning_v0/model" # only used if training_mode == "fine_tuning"

experiment_name = f'{model_name}_no_preprocessing_{training_mode}_v'

dir_index = 0
while os.path.isdir(os.path.join(os.getcwd(), experiment_name + str(dir_index))):
  dir_index += 1

os.mkdir(os.path.join(os.getcwd(), experiment_name + str(dir_index)))

experiment_name += str(dir_index)

print("Set experiment name")

# #create an experiment with your api key
experiment = comet_ml.Experiment(
    api_key="LWf0Xh0BnuzPNtCkqvUJz2mKc",
    project_name=experiment_name,
    auto_histogram_weight_logging=True,
    auto_histogram_gradient_logging=True,
    auto_histogram_activation_logging=True,
)

print("Set experiment params")

from tensorflow import keras

print("Imported keras lib")

# Define a function for decaying the learning rate.
# You can define any decay function you need.
def decay(epoch):
  if epoch < 15:
    return 1e-3
  elif epoch >= 15 and epoch < 80:
    return 1e-4
  else:
    return 1e-5
  
print("Defined custom lr decay function")

#these will all get logged
params = {
  "batch_size": 16,
  "epochs": 200,
  "input_size": (224, 224, 3),
  "learning_rate": 1e-3,
  "lr_scheduler": decay
}

print("Initialized models params")

from dataloader import *

print("Imported dataloader file")

# the data, shuffled and split between train and test sets
datasets_paths = ["/home/guilherme/Downloads/breast_20x", "/home/guilherme/Downloads/mouth_20x"]

print("Set path to datasets")

import albumentations as A

transformations_list = [
  A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=0.5),
  A.ColorJitter(brightness=(0.1, 0.5), contrast=(0.1, 0.5), saturation=(0.1, 0.5), hue=(-0.2, 0.2), p=0.5),
  A.ShiftScaleRotate(shift_limit=(0.05, 0.2), scale_limit=(-0.1, 0.5), rotate_limit=45, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_WRAP, p=0.5)
]

print("Defined list of transformations")

img_paths = scan_datasets(datasets_paths)
pacients = retrieve_pacients(img_paths)

print(f"Retrieved {len(pacients)} pacients")

folds = create_folds(pacients, img_paths, 5)

print(f"Created {len(folds)} folds for Cross Validation")

upsampling_factor = 5
upsampled_folds = upsample_folds(folds, upsampling_factor=upsampling_factor, transformations_list=transformations_list)

print(f"Upsampled the folds x{upsampling_factor} times")

policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

mirrored_strategy = tf.distribute.MirroredStrategy()
print("Set the strategy to mirrored")

for train, test in PermuteFolds(folds=upsampled_folds):
  
  train_loader = CustomDataGenerator(
                 data=train,
                 batch_size=16,
                 input_size=params["input_size"],
                 shuffle=False,
                 normalize=False)

  print("Initialized train loader")

  test_loader = CustomDataGenerator(
                 data=test,
                 batch_size=16,
                 input_size=params["input_size"],
                 shuffle=False,
                 normalize=False)

  print("Initialized test loader")

  from model import get_model

  with mirrored_strategy.scope():
    custom_model = get_model(
                      base_model_name=model_name, 
                      params=params, 
                      training_mode=training_mode, 
                      weight_path=weight_path,
                      saved_model_path=saved_model_path
                    )

  # Define the checkpoint directory to store the checkpoints.
  checkpoint_dir = os.path.join(os.getcwd(), experiment_name, "checkpoints")
  # Define the name of the checkpoint files.
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_")

  print("Set the save dir for training")

  # Put all the callbacks together.
  callbacks = [
      keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=20, verbose=1, mode='auto'),
      keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                        save_weights_only=True,
                                        save_best_only=True),
      keras.callbacks.LearningRateScheduler(params["lr_scheduler"]),
      keras.callbacks.TensorBoard(log_dir=os.path.join(os.getcwd(), experiment_name, "logs"))
  ]

  print("Joined all callbacks into a list")

  import sklearn

  print("Imported scikit learn library for Cross-Validation")

  #will log metrics with the prefix 'train_'
  with experiment.train():
    history = custom_model.fit(
                        train_loader,
                        batch_size=params["batch_size"],
                        epochs=params["epochs"],
                        verbose=1,
                        validation_data=test_loader,
                        callbacks=callbacks)

  #will log metrics with the prefix 'test_'
  with experiment.test():
    loss, accuracy = custom_model.evaluate(test_loader)
    metrics = {
        'loss':loss,
        'accuracy':accuracy
    }

  os.mkdir(os.path.join(os.getcwd(), experiment_name, "model"))
  custom_model.save(os.path.join(os.getcwd(), experiment_name, "model"), save_format="tf")

  experiment.log_metrics(metrics)

  experiment.log_parameters(params)

  experiment.end()