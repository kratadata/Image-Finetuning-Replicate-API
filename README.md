# Ostris Image Fine-Tuning

A tool for fine-tuning image models using Replicate's API.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/kratadata/Image-Finetuning-Replicate-API
```

2. Open the folder in your IDE and open the terminal in the folder.

3. Set up a virtual environment:

```bash
pip install virtualenv
python -m venv ostris
source ostris/bin/activate
```

4. Install Replicate library:

```bash
pip install replicate
```

## Usage

### Training

**IMPORTANT:** in config.json, make sure to set the correct variables:
- replicate_api_token: your Replicate API token
- dataset_path: path to your training images (e.x. dataset/)
- trigger_word: the word that will be used to trigger the model (e.x. PERSONAAAA)
- output_destination: the destination of the fine-tuned model (e.x. kratadata/group1)

```bash
python train.py # training a new model, logs will be saved in logs/
```


### Cancelling

```bash
python cancel.py --list # list all training jobs
python cancel.py --id <training_id> # cancel a specific training job, you can find the training id in the logs/ directory
```

