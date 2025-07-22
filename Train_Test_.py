import os
import pandas as pd
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
from math import ceil

# ==============================================================
# CHOOSE THE BASE MODEL – Uncomment ONE line only
# ==============================================================
from tensorflow.keras.applications import ResNet50        as BaseModel
# from tensorflow.keras.applications import MobileNetV2      as BaseModel
# from tensorflow.keras.applications import DenseNet121      as BaseModel
# from tensorflow.keras.applications import EfficientNetB0   as BaseModel
# from tensorflow.keras.applications import EfficientNetV2B3  as BaseModel

# Grab the matching preprocessing function
from importlib import import_module
preprocess_input = import_module(BaseModel.__module__).preprocess_input

# Set numpy print options
np.set_printoptions(precision=6, suppress=True)
pd.set_option('display.precision', 6)

train_folder = 'GOOD_QUALITY_TRAIN'             # Select the folder with the images for training
validation_folder = 'Validation_Images'
p_validation_folder = 'P_Validation_Images'
validation_csv = 'validation_data_classes.csv'
p_validation_csv = 'P_validation_data_classes.csv'

# Define parameters
image_height, image_width = 240, 320  # original size of each image
n_eps = 40  # number of epochs per model
imgs_per_batch = 64
n_models = 2 # number of models trained

# Generate all training and testing data
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=(image_height, image_width),
    batch_size=imgs_per_batch,
    class_mode='binary',
    shuffle=True,
)

num_train_images = sum([len(files) for r, d, files in os.walk(train_folder)])
train_steps_per_epoch = int(np.ceil(num_train_images / imgs_per_batch))

# Load and process validation data
validation_annotations = pd.read_csv(validation_csv)
validation_annotations.columns = validation_annotations.columns.str.strip()
validation_annotations['Fish'] = validation_annotations['Fish'].astype(str)
validation_annotations = validation_annotations.sort_values('filename')

p_validation_annotations = pd.read_csv(p_validation_csv)
p_validation_annotations.columns = p_validation_annotations.columns.str.strip()
p_validation_annotations['Fish'] = p_validation_annotations['Fish'].astype(str)
p_validation_annotations = p_validation_annotations.sort_values('filename')

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validation_annotations,
    directory=validation_folder,
    x_col='filename', 
    y_col='Fish',
    target_size=(image_height, image_width),
    batch_size=imgs_per_batch,
    class_mode='binary',
    shuffle=False,
)
validation_steps = int(np.ceil(validation_annotations.shape[0] / imgs_per_batch))

p_validation_generator = val_datagen.flow_from_dataframe(
    dataframe=p_validation_annotations,
    directory=p_validation_folder,
    x_col='filename',
    y_col='Fish',
    target_size=(image_height, image_width),
    batch_size=imgs_per_batch,
    class_mode='binary',
    shuffle=False,
)
p_validation_steps = int(np.ceil(p_validation_annotations.shape[0] / imgs_per_batch))

# Convert CSV "Fish" column to int arrays for manual label referencing
val_true_full = validation_annotations["Fish"].astype(int).values
p_val_true_full = p_validation_annotations["Fish"].astype(int).values

for model_num in range(1, n_models+1):
    inputs = Input(shape=(image_height, image_width, 3))
    base_model = BaseModel(weights='imagenet', include_top=False, input_tensor=inputs)

    print(f"\nSelected base model: {BaseModel.__name__}")
    total_layers = len(base_model.layers)
    print(f"Number of layers in the base model: {total_layers}")

    n_layers_unf = ceil(total_layers/2)
    layers_to_unfreeze = list(range(0, n_layers_unf))
    print(f"Number of layers to unfreeze: {n_layers_unf}")

    for layer in base_model.layers:
        layer.trainable = False

    for index in layers_to_unfreeze:
        base_model.layers[-(index + 1)].trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)

    checkpoint_filepath = f'best_model_based_on_good_quality_test_accuracy_{model_num}.keras'
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        mode='max'
    )

    model.compile(
        optimizer=Adam(learning_rate=2e-6, amsgrad=True),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Lists to store overall metrics
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    p_val_accuracies = []
    p_val_losses = []
    elapsed_times = []

    start_time = time.time()
    elapsed_time = 0
    
    for epoch in range(n_eps):
        print(f"Model {model_num}/{n_models} - Epoch {epoch+1}/{n_eps}", flush=True)

        # Train for one epoch
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=1,
            verbose=0,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[model_checkpoint]
        )

        # Retrieve metrics
        train_acc = history.history['accuracy'][0]
        train_loss = history.history['loss'][0]
        val_acc = history.history['val_accuracy'][0]
        val_loss = history.history['val_loss'][0]

        # Evaluate on P-validation set
        p_val_loss, p_val_acc = model.evaluate(
            p_validation_generator,
            steps=p_validation_steps,
            verbose=0
        )

        # Record metrics
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        p_val_accuracies.append(p_val_acc)
        p_val_losses.append(p_val_loss)

        # Measure time
        t_now = time.time()
        epoch_time = t_now - start_time - elapsed_time
        elapsed_time = t_now - start_time
        elapsed_times.append(elapsed_time)

        # Print epoch results
        print(f"Training set: accuracy = {train_acc:.4f}, loss = {train_loss:.4f}", flush=True)
        print(f"Test set (good quality): good_quality_test_accuracy = {val_acc:.4f}, good_quality_test_loss = {val_loss:.4f}", flush=True)
        print(f"Test set (poor quality): poor_quality_test_accuracy = {p_val_acc:.4f}, poor_quality_test_loss = {p_val_loss:.4f}", flush=True)

        print(f"Epoch Time: {epoch_time:.2f} seconds", flush=True)
        print(f"Training Time: {elapsed_time:.2f} seconds", flush=True)

        if val_acc > max(val_accuracies[:-1], default=-float("inf")):
            print(f"good_quality_test_accuracy improved to {val_acc:.5f}, saving model to {checkpoint_filepath}", flush=True)
        else:
            best_so_far = max(val_accuracies[:-1], default=0)
            print(f"good_quality_test_accuracy did not improve from {best_so_far:.5f}", flush=True)
        print("\n", flush=True)

    # Save the best model
    model.save(checkpoint_filepath)

    # Create DataFrame with all metrics
    metrics_df = pd.DataFrame({
        "Epoch": range(1, n_eps + 1),
        f"Train_Accuracy_{model_num}": train_accuracies,
        f"Train_Loss_{model_num}": train_losses,
        f"good_quality_test_accuracy_{model_num}": val_accuracies,
        f"good_quality_test_loss_{model_num}": val_losses,
        f"poor_quality_test_accuracy_{model_num}": p_val_accuracies,
        f"poor_quality_test_loss_{model_num}": p_val_losses,
        f"Elapsed_Time_{model_num}": elapsed_times,
    })

    metrics_csv_filename = f"accuracy_loss_metrics_{model_num}.csv"
    metrics_df.to_csv(metrics_csv_filename, index=False)
    print(f"Metrics saved to {metrics_csv_filename}")

    # Clear session and delete model to release memory
    tf.keras.backend.clear_session()
    del model
    del base_model