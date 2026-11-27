
import tensorflow as tf
import json
import os

MODEL_PATH = "models/isl_mobilenetv2.keras"
CLASSES_PATH = "models/classes.json"

print("\n========== LOADING MODEL ==========")

model = tf.keras.models.load_model(MODEL_PATH)

print("\n========== MODEL ARCHITECTURE ==========")

model.summary()

print("\n========== MODEL PARAMETERS ==========")

total_params = model.count_params()

trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)

print("Total Parameters:", f"{total_params:,}")
print("Trainable Parameters:", f"{trainable_params:,}")
print("Non-trainable Parameters:", f"{non_trainable_params:,}")

print("\n========== MODEL INPUT/OUTPUT ==========")

print("Input Shape:", model.input_shape)
print("Output Shape:", model.output_shape)

print("\n========== MODEL SIZE ==========")

size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print(f"Model Size: {size_mb:.2f} MB")

print("\n========== CLASS INFORMATION ==========")

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

if isinstance(classes, dict):
    class_names = list(classes.keys())
else:
    class_names = classes

print("Number of Classes:", len(class_names))
print("Classes:", class_names)

print("\n========== COMPILATION INFORMATION ==========")

print("Optimizer:", model.optimizer)

if model.optimizer is not None:
    print("Loss Function:", model.loss)
    print("Metrics:", model.metrics)

print("\nModel inspection completed.")
