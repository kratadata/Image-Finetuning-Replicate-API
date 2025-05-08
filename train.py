import replicate
import os
from zipfile import ZipFile
import json
from pprint import pprint
from datetime import datetime
from logProgess import print_training, log_training


def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)
    
config = load_config()
dataset_path = config["dataset_path"]
trigger = config["trigger_word"]
dest = config["output_destination"]
api_token = config["replicate_api_token"]

os.environ["REPLICATE_API_TOKEN"] = api_token

def dataset_to_replicate(dataset_path):
    zip_filename = f"{os.path.basename(dataset_path)}.zip"
    with ZipFile(zip_filename, 'w') as zip_object:
        for folder_path, _, filenames in os.walk(dataset_path):
            for filename in filenames:
                # Create filepath of files in directory
                file_path = os.path.join(folder_path, filename)
                # Add files to zip file - preserve relative path within the dataset
                arcname = os.path.relpath(file_path, start=os.path.dirname(dataset_path))
                zip_object.write(file_path, arcname)
    
    return zip_filename

def train(dataset_path):
    try:
        zip_file = dataset_to_replicate(dataset_path)
        training = replicate.trainings.create(
        destination= dest,
        version="ostris/flux-dev-lora-trainer:c6e78d2501e8088876e99ef21e4460d0dc121af7a4b786b9a4c2d75c620e300d",
        input={
            "steps": 1000,
            "lora_rank": 16,
            "optimizer": "adamw8bit",
            "batch_size": 1,
            "resolution": "512,768,1024",
            "autocaption": True,
            "input_images": open(zip_file, "rb"), 
            "trigger_word": trigger,
            "learning_rate": 0.0004,
            "wandb_project": "flux_train_replicate",
            "wandb_save_interval": 100,
            "caption_dropout_rate": 0.05,
            "cache_latents_to_disk": False,
            "wandb_sample_interval": 100,
            "gradient_checkpointing": False
        },
        )
        return training
    except Exception as e:
        print(e)
  

if __name__ == "__main__":
    if "REPLICATE_API_TOKEN" not in os.environ:
        print("ERROR: REPLICATE_API_TOKEN environment variable is not set.")
        print("Please set it with: export REPLICATE_API_TOKEN=your_token_here")
        exit(1)
    
    training = train(dataset_path)
    
    if training:
        print_training(training)
        # Create log file with timestamp and training ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = f"logs/{timestamp}_{training.id}.json"
        log_training(training, log_file_path)
        
    else:
        print("Training failed to start")