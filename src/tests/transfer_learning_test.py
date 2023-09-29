import shutil
import os
import glob
import numpy as np
from keras.models import load_model
from pathlib import Path
from PIL import Image
from src.DenseDepth.layers import BilinearUpSampling2D
from src.DenseDepth.utils import predict, load_images
from src.tests.performance import performance_metrics

base_dir = Path.cwd()
input_dir = f"{base_dir}/img/input/"
output_dir = f"{base_dir}/img/output/transfer_learning_test"
tmp_dir = f"{base_dir}/img/tmp/"
dataset_dir = f"{base_dir}/dataset/"
model_path = f"{base_dir}/models/nyu.h5"


def copy_dataset_to_input_dir():
    clean_input_dir()
    dataset_images = os.listdir(dataset_dir)
    print("Copy dataset to input directory")
    for image in dataset_images:
        shutil.copyfile(f"{dataset_dir}{image}", f"{input_dir}{image}")
        print(f"\tCopying {image} to input directory...")


def resize_images():
    clean_tmp_dir()
    images = os.listdir(input_dir)
    print("Resize dataset to 640x480")
    for image in images:
        filename = image.split(".")[0]
        original_image = Image.open(f"{input_dir}{image}")
        resized_image = original_image.resize((640, 480))
        resized_image.save(f"{tmp_dir}{filename}.png")
        print(f"\tResizing {image}...")


def clean_input_dir():
    input_files = os.listdir(tmp_dir)
    for file in input_files:
        os.remove(f"{tmp_dir}{file}")
    print("Input directory cleaned")


def clean_tmp_dir():
    tmp_files = os.listdir(tmp_dir)
    for file in tmp_files:
        os.remove(f"{tmp_dir}{file}")
    print("Tmp directory cleaned")


@performance_metrics
def estimate_monocular_depth():
    # Custom object needed for inference and training
    custom_objects = {
        "BilinearUpSampling2D": BilinearUpSampling2D,
        "depth_loss_function": None,
    }

    # Load model into GPU / CPU
    print("Loading model...")
    print(model_path)
    model = load_model(model_path, custom_objects=custom_objects, compile=False)
    print("\nModel loaded ({0}).".format(model_path))

    # Input images
    inputs = load_images(glob.glob("/img/tmp/*.png"))
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
        img.save(output_dir + str(i) + ".png")
        saved_images += 1
        print(f"Saved ({saved_images}) images of ({ total_images }) dataset")


def run_test():
    print("Transfer Learning test started...")
    copy_dataset_to_input_dir()
    resize_images()
    estimate_monocular_depth()
    clean_tmp_dir()
    clean_input_dir()
    print("Transfer Learning test executed successfully...")
