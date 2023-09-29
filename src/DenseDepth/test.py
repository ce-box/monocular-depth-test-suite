import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.python.keras.layers import (
    Layer,
    InputSpec,
)  # Para que lo detecten las nuevas versiones de TF
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

from PIL import Image
import numpy as np


# Argument Parser
parser = argparse.ArgumentParser(
    description="High Quality Monocular Depth Estimation via Transfer Learning"
)
parser.add_argument(
    "--model", default="/models/nyu.h5", type=str, help="Trained Keras model file."
)
parser.add_argument(
    "--input", default="/img/tmp/*.png", type=str, help="Input filename or folder."
)

# Custom object needed for inference and training
custom_objects = {
    "BilinearUpSampling2D": BilinearUpSampling2D,
    "depth_loss_function": None,
}


def estimate_monocular_depth():
    args = parser.parse_args()

    # Load model into GPU / CPU
    print("Loading model...")
    model = load_model(args.model, custom_objects=custom_objects, compile=False)
    print("\nModel loaded ({0}).".format(args.model))

    # Input images
    inputs = load_images(glob.glob(args.input))
    print(
        "\nLoaded ({0}) images of size {1}.".format(inputs.shape[0], inputs.shape[1:])
    )

    # Compute results
    outputs = predict(model, inputs)
    print("Outputs predicted...")

    # Extended test

    [images, width, height, dimension] = outputs.shape

    saved_images, total_images = 0, len(images)
    for i in range(images):
        processed_image = outputs[i, :, :, :]
        i8 = (
            (
                (processed_image - processed_image.min())
                / (processed_image.max() - processed_image.min())
            )
            * 255.9
        ).astype(np.uint8)
        i8_s = np.squeeze(i8)
        img = Image.fromarray(i8_s)
        img.save("/img/output/transfer_learning_test/" + str(i) + ".png")
        saved_images += 1
        print(f"Saved ({saved_images}) images of ({ total_images }) dataset")
