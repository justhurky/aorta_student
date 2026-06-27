"""
Main entry point for the medical DICOM processing pipeline.
Orchestrates extraction, conversion, training, and professional prediction.
"""

import argparse
import sys
import os

# Internal modules
from utils.logger_setup import setup_logger
from data_prep_scripts.extracted_frames import FrameExtractor
from data_prep_scripts.json_to_txt import JsonToTxtConverter
from data.train.model_training import ModelTrainer
from data.prediction.model_predictor import ModelPredictor

def main():
    """
    Main execution loop. Handles CLI arguments and module dispatching.
    """
    logger = setup_logger()

    parser = argparse.ArgumentParser(
        description="Medical Imaging: YOLO-based Measurement and DICOM Analytics"
    )
    
    parser.add_argument(
        "command", 
        choices=["extract", "convert", "train", "predict"], 
        help="Pipeline step to execute."
    )

    # Path & Meta arguments
    parser.add_argument("--input", help="Source file or directory path.")
    parser.add_argument("--output", help="Target output directory.")
    parser.add_argument("--dicom", help="Reference DICOM file for mm conversion.")
    parser.add_argument("--model", default="yolov8n.pt", help="Weights file (.pt).")
    parser.add_argument("--conf", type=float, default=0.45, help="Confidence threshold (default 0.45).")
    
    parser.add_argument(
        "--mode",
        choices=["batch", "single", "video"],
        default="batch",
        help="Inference mode (default: batch)."
    )

    args = parser.parse_args()

    # --- CLI DISPATCHER ---

    if args.command == "extract":
        logger.info("Command [EXTRACT] initiated.")
        input_root = args.input or "dataset_preparation/raw_dicom"
        output_root = args.output or "dataset_preparation/extracted_frames"
        try:
            extractor = FrameExtractor(logger)
            extractor.process_all_dicoms(input_root, output_root)
            logger.info("Extraction completed.")
        except Exception as e:
            logger.critical(f"Extraction failed: {e}")
            sys.exit(1)

    elif args.command == "convert":
        logger.info("Command [CONVERT] initiated.")
        input_root = args.input or "dataset_preparation/annotations_json"
        output_root = args.output or "dataset_preparation/annotations_txt"
        try:
            converter = JsonToTxtConverter(logger)
            converter.process_all(input_root, output_root)
            logger.info("Conversion completed.")
        except Exception as e:
            logger.critical(f"Conversion failed: {e}")
            sys.exit(1)

    elif args.command == "train":
        logger.info(f"Command [TRAIN] initiated for {args.model}.")
        data_yaml = args.input or "data/data.yaml"
        try:
            trainer = ModelTrainer(logger)
            trainer.run_train(model_variant=args.model, data_yaml=data_yaml)
            logger.info("Training completed.")
        except Exception as e:
            logger.critical(f"Training failed: {e}")
            sys.exit(1)

    elif args.command == "predict":
        logger.info(f"Command [PREDICT] initiated (Mode: {args.mode}).")
        
        # Select best weights if not specified
        model_path = args.model if args.model.endswith(".pt") else "runs/detect/train/weights/best.pt"
        output_root = args.output or "data/prediction"
        
        try:
            predictor = ModelPredictor(logger)
            
            if args.mode == "single":
                if not args.input: raise ValueError("--input is required for single mode.")
                predictor.run_single_prediction(model_path, args.input, output_root, dicom_ref=args.dicom, conf=args.conf)
                
            elif args.mode == "video":
                if not args.input: raise ValueError("--input is required for video mode.")
                predictor.run_video_prediction(model_path, args.input, output_root, dicom_ref=args.dicom, conf=args.conf)
                
            else: # Default 'batch'
                source = args.input or "data/dataset/test_list.txt"
                predictor.run_prediction(model_path, source, output_root, dicom_ref=args.dicom, conf=args.conf)
                
            logger.info("Measurement and prediction pipeline finished successfully.")
            
        except Exception as e:
            logger.critical(f"Prediction pipeline error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()
