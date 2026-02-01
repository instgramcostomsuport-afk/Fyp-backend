import tensorflow as tf

# Load the model
model = tf.keras.models.load_model(r"models\nutrifoodnet_final.h5", compile=False)

# Confirm it loaded
print("MODEL LOADED")
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Number of classes:", model.output_shape[-1])
