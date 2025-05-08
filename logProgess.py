import os
import json
import time
from datetime import datetime
import replicate
from pprint import pprint

def print_training(training):
    """Print detailed information about the training object"""
    print("Training Model:", training.model)
    print("Training Model Version:", training.version)
    print("Training ID:", training.id)
    print("Training Status:", training.status)
    print("Training URL:", training.urls.get("get"))
    print("\nTraining Configuration:")
    pprint(training.input)

def log_training(training, log_file_path):
    """
    Log training information to a file
    Logs status and logs every 5 seconds, and timestamps once
    """
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    
    # Initialize log data
    log_data = {
        "id": training.id,
        "model": training.model,
        "version": training.version,
        "created_at": None,
        "status_history": [],
        "completed_at": None

    }
    
    # Save created_at timestamp once
    log_data["created_at"] = training.created_at
    
    print(f"Logging training information to {log_file_path}")
    print("Training Model:", training.model)
    print("Training Version:", training.version)
    print(f"Training ID: {training.id}")
    print(f"Created at: {training.created_at}")
    
    try:
        # Continue logging until training is completed or failed
        while training.status not in ["succeeded", "failed", "canceled"]:
            # Get current status and logs
            training = replicate.trainings.get(training.id)
            current_time = datetime.now().isoformat()
            
            # Add status to history
            log_data["status_history"].append({
                "timestamp": current_time,
                "status": training.status
            })
            
            # Save log data to file
            with open(log_file_path, 'w') as f:
                json.dump(log_data, f, indent=2)
            
            print(f"Status: {training.status} - {current_time}")
            time.sleep(5)
        
        # Training completed, save final status and completed_at
        log_data["completed_at"] = training.completed_at if hasattr(training, "completed_at") else datetime.now().isoformat()
        log_data["status_history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": training.status
        })
        
        #save
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Training completed with status: {training.status}")
        print(f"Completed at: {log_data['completed_at']}")
        
    except Exception as e:
        print(f"Error while logging: {e}")
        with open(log_file_path, 'w') as f:
            json.dump(log_data, f, indent=2)