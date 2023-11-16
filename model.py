import tensorflow as tf
from tensorflow import keras
from keras.applications import ResNet50, DenseNet121, EfficientNetB0

print("Imported TF and models")

def get_model(base_model_name: str, params: dict, training_mode: str, weight_path: str) -> tf.keras.Model:
    
    trainable = True # boolean to set layers trainability
    
    # Type of training
    if training_mode == "transfer_learning":
        trainable = False
        weight = "imagenet"
    elif training_mode == "fine_tuning":
        weight = None
    elif training_mode == "from_scratch":
        weight = None
    
    # Choose model
    if base_model_name == "resnet":
        base_model = ResNet50(
                        include_top=False,
                        weights=weight,
                        pooling="avg"
                    )
    elif base_model_name == "densenet":
        base_model = DenseNet121(
                        include_top=False,
                        weights=weight,
                        input_tensor=None,
                        input_shape=None,
                        pooling="avg"
                    )
    elif base_model_name == "efficientnet":
        base_model = EfficientNetB0(
                        include_top=False,
                        weights=weight,
                        input_tensor=None,
                        input_shape=None,
                        pooling="avg"
                    )
    else:
        print("Error with model name")
        quit()

    print(f"Initialized {base_model_name} model")
    
    base_model.trainable = trainable
    
    if training_mode == "fine_tuning":
        base_model.load_weights(weight_path) #! Temp workaround to load weights
        for layer in base_model.layers:
            if "bn" in layer.name:
                layer.trainable = False
        print("Froze all BN layers in the model")
    
    print(f"Set base model layers training to {trainable}")
        
    inputs = keras.Input(shape=params["input_size"])
    x = base_model(inputs, training=False)
    
    # Dropout layer
    x = keras.layers.Dropout(rate=0.5)(x)
    
    # A Dense classifier with a single unit (binary classification)
    outputs = keras.layers.Dense(units=1, activation="sigmoid")(x)
    
    model = keras.Model(inputs, outputs)
    
    print(f"Added {base_model_name} into custom model")
    
    #print model.summary() to preserve automatically in `Output` tab
    print(model.summary())
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), 
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy()]
    )
    
    print("Compiled custom model")
    
    return model