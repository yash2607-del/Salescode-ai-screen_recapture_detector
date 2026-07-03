"""
Main entry point for training the screen-recapture-detector model.
Triggers the two-stage transfer learning pipeline.
"""
from src.trainer import Trainer

def main():
    """
    Instantiates and runs the training pipeline.
    """
    print("Starting screen-recapture-detector training pipeline...")
    trainer = Trainer()
    trainer.load_datasets()
    trainer.run_pipeline()
    print("Training pipeline finished successfully!")

if __name__ == "__main__":
    main()
