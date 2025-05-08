import replicate
import sys
import os
import argparse
import json
from datetime import datetime

def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

config = load_config()
api_token = config["replicate_api_token"]
os.environ["REPLICATE_API_TOKEN"] = api_token

def cancel_training(training_id):
    """
    Cancel a training job by ID and log the cancellation
    """
    try:
        # Get the training object
        training = replicate.trainings.get(training_id)
        
        # Print current status
        print(f"Training ID: {training.id}")
        print(f"Current status: {training.status}")
        
        # Only cancel if not already completed
        if training.status in ["succeeded", "failed", "canceled"]:
            print(f"Training already in terminal state: {training.status}")
            return False
        
        # Cancel the training
        training.cancel()
        
        # Get updated training object
        updated_training = replicate.trainings.get(training_id)
        print(f"Training canceled.")
        
        return True
    
    except Exception as e:
        print(f"Error canceling training: {e}")
        return False

def list_active_trainings():
    """
    List all active training jobs
    """
    try:
        # Get all trainings
        trainings = replicate.trainings.list()
        
        # Filter for active trainings
        active_trainings = [t for t in trainings if t.status not in ["succeeded", "failed", "canceled"]]
        
        if not active_trainings:
            print("No active training jobs found.")
            return
        
        print(f"Found {len(active_trainings)} active training jobs:")
        for idx, training in enumerate(active_trainings, 1):
            print(f"{idx}. ID: {training.id}")
            print(f"   Status: {training.status}")
            print(f"   Created: {training.created_at}")
            print(f"   Model: {training.destination}")
            print()
            
    except Exception as e:
        print(f"Error listing trainings: {e}")

if __name__ == "__main__":
     # Set up argument parser
    parser = argparse.ArgumentParser(description="Cancel Replicate training")
    parser.add_argument("--id", help="Training ID to cancel")
    parser.add_argument("--list", action="store_true", help="List active training jobs")
    
    args = parser.parse_args()
    
    # List active trainings if requested
    if args.list:
        list_active_trainings()
        sys.exit(0)
    
    # Cancel specific training if ID provided
    if args.id:
        success = cancel_training(args.id)
        sys.exit(0 if success else 1)
    
     # If no arguments provided, show help
    if not (args.id or args.list):
        parser.print_help()
        print("\nExample usage:")
        print("  python cancel_train.py --list")
        print("  python cancel_train.py --id r8eqrj3rpxdtvmbkxsmhqt5f4e")