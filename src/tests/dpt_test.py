import shutil, os
from pathlib import Path

from src.DPT.run_monodepth import run
from src.tests.performance import performance_metrics

# base directory
base_dir = Path.cwd()

# working directories
input_dir = f"{base_dir}/img/input/"
output_dir = f"{base_dir}/img/output/dpt_test"
tmp_dir = f"{base_dir}/img/tmp/"

# dataset paths
dataset_dir = f"{base_dir}/dataset/"

# models
default_models = {
    "midas_v21": f"{base_dir}/src/DPT/weights/midas_v21-f6b98070.pt",
    "dpt_large": f"{base_dir}/src/DPT/weights/dpt_large-midas-2f21e586.pt",
    "dpt_hybrid": f"{base_dir}/src/DPT/weights/dpt_hybrid-midas-501f0c75.pt",
    "dpt_hybrid_kitti": f"{base_dir}/src/DPT/weights/dpt_hybrid_kitti-cb926ef4.pt",
    "dpt_hybrid_nyu": f"{base_dir}/src/DPT/weights/dpt_hybrid_nyu-2ce69ec7.pt",
}


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
def estimate_monocular_depth_dpt():
    run(input_dir, output_dir, default_models["dpt_hybrid"], "dpt_hybrid", True)


def run_test():
    print("DPT test started...")
    copy_dataset_to_input_dir()
    estimate_monocular_depth_dpt()
    clean_input_dir()
    print("DPT test executed successfully...")


if __name__ == "__main__":
    run_test()
