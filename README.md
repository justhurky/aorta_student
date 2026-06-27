# Medical Imaging: YOLO-based Measurement and DICOM Analytics

This project is a professional medical imaging pipeline designed to process DICOM files, extract frames, manage annotations, and perform distance measurements using YOLO object detection.

## Project Structure

- `data/`: Core data management.
  - `dataset/`: Training, validation, and test datasets.
  - `prediction/`: Inference results and scripts.
  - `train/`: Training scripts and YAML configuration.
- `data_prep_scripts/`: Utilities for preparing the data.
  - `extracted_frames.py`: Converts DICOMs to flat image collections.
  - `json_to_txt.py`: Converts VoTT JSON labels to YOLO format.
- `utils/`: Common utilities like logging.
- `main.py`: The central orchestrator for all pipeline steps.
- `best.pt`: Optional trained model weights. Model files are intentionally not committed.

## Installation

```bash
pip install -r requirements.txt
```

## Usage Instructions

### 1. Extract Frames from DICOM
Converts all DICOM files in a source directory into a flat collection of JPG images.
```bash
python main.py extract --input "data/dataset/raw_dicom" --output "data/dataset/all_frames"
```

### 2. Convert Annotations
Converts VoTT JSON exports to YOLO TXT format.
```bash
python main.py convert --input "path/to/json/folder" --output "data/dataset/labels"
```

### 3. Train the Model
Trains a YOLO model using the prepared dataset. Results are saved in the `runs/` directory.
```bash
python main.py train --model yolov8n.pt --input "data/data.yaml"
```

### 4. Run Prediction (Inference)
Performs detection and measures horizontal distances. Generates annotated images and a PDF report.

**Single Image:**
```bash
python main.py predict --mode single --input "data/dataset/test/images/frame_0010.jpg" --model best.pt --dicom "data/dataset/raw_dicom/A0009 (PW poststent hyperemia from LAT) másolata"
```

**Batch (Directory):**
```bash
python main.py predict --mode batch --input "data/dataset/test/images" --model best.pt --dicom "data/dataset/raw_dicom/A0009 (PW poststent hyperemia from LAT) másolata"
```

**Video:**
```bash
python main.py predict --mode video --input "data/dataset/test/videos/sample.mp4" --model best.pt
```

## Kronosz HPC Setup
To run the training on the Kronosz server, ensure all dependencies are installed and use the `train` command. The results will be stored in the `runs/detect/train/` folder.

## Git Ignore
The following folders are excluded from version control:
- datasets and DICOM files
- trained model weights
- runtime logs, reports, predictions, and training runs
