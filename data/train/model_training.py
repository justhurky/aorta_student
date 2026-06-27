"""
Module for training the YOLO model in a medical imaging context.
Uses pre-partitioned folders (80-10-10) for training.
"""

from ultralytics import YOLO

class ModelTrainer:
    """
    A class to manage the training lifecycle of YOLO models 
    using a pre-organized folder structure.
    """

    def __init__(self, logger):
        """
        Initializes the ModelTrainer with a centralized logger.

        Args:
            logger (logging.Logger): Logger instance for status reporting.
        """
        self.logger = logger

    def run_train(self, model_variant, data_yaml, epochs=125, dataset_path=None):
        """
        Executes the YOLO training process using the provided YAML configuration.

        Args:
            model_variant (str): The YOLO model variant (e.g., 'yolov8m.pt').
            data_yaml (str): Path to the training configuration file (data.yaml).
            epochs (int): Number of training iterations.
        """
        try:
            # Step 1: Initialize model
            self.logger.info(f"Initializing model: {model_variant}")
            model = YOLO(model_variant)

            # Step 2: Start training
            # Note: workers=0 is set for Windows stability as discussed before
            self.logger.info(f"Starting training with {data_yaml} for {epochs} epochs...")
            
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=640,
                plots=True,
                device=0,      
                batch=4,       
                workers=2,     
                amp=False      
            )
            
            self.logger.info("Training process finished successfully.")

        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise