import cv2
import torch
import shutil
import os

from pathlib import Path
from PIL import Image

from src.tests.performance import performance_metrics

# base directory
base_dir = Path.cwd()

# working directories
input_dir = f"{base_dir}/img/input/"
output_dir = f"{base_dir}/img/output/mixing2_test"
tmp_dir = f"{base_dir}/img/tmp/"

# dataset paths
dataset_dir = f"{base_dir}/dataset/"

# models
models = {
    "large": "DPT_Large",  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    "hybrid": "DPT_Hybrid",  # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    "small": "MiDaS_small",  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
}

device = None
transform = None
midas = None


def model_setup():
    model_type = models["large"]

    global midas, device, transform
    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform


def estimate_monocular_depth(filename, output_path):
    global midas, device, transform
    # Read image
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Transform image and send it to CPU/GPU
    input_batch = transform(img).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Store result in output folder
    output = prediction.cpu().numpy()

    # Save image
    image_fullname = filename.split("/")[-1]
    image_filename = image_fullname.split(".")[0]
    depth_image = Image.fromarray(output)
    output_image = depth_image.save(f"{output_path}/{image_filename}_depth.tif")


def copy_dataset_to_input_dir():
    clean_input_dir()
    dataset_images = os.listdir(dataset_dir)
    print("Copy dataset to input directory")
    for image in dataset_images:
        shutil.copyfile(f"{dataset_dir}{image}", f"{input_dir}{image}")
        print(f"\tCopying {image} to input directory...")


def clean_input_dir():
    input_files = os.listdir(tmp_dir)
    for file in input_files:
        os.remove(f"{tmp_dir}{file}")
    print("Input directory cleaned")


@performance_metrics
def estimate_monocular_depth_m2():
    images = os.listdir(input_dir)
    for image in images:
        estimate_monocular_depth(f"{input_dir}{image}", output_dir)


def run_test():
    print("Mixing 2 test started...")
    copy_dataset_to_input_dir()
    model_setup()
    estimate_monocular_depth_m2()
    clean_input_dir()
    print("Mixing 2 test executed successfully...")


if __name__ == "__main__":
    run_test()
