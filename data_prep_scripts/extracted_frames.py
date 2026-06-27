import os
import cv2
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

class FrameExtractor:
    """
    A class to handle the extraction and processing of frames from DICOM files.

    This class manages the full workflow of reading DICOM files, applying 
    necessary medical imaging transformations, and saving the resulting 
    frames into standard image formats.
    """

    def __init__(self, logger):
        """
        Initializes the FrameExtractor with a logger instance.

        Args:
            logger (logging.Logger): A logger object used for tracking errors and process information.
        """
        self.logger = logger

    def apply_windowing(self, frame, dicom_data):
        """
        Applies contrast settings embedded in the DICOM file.

        The function uses pydicom's apply_voi_lut to adjust the Window Center
        and Window Width values on the frame to ensure soft tissues are visible.

        Args:
            frame (numpy.ndarray): Matrix containing the raw pixel data.
            dicom_data (pydicom.dataset.FileDataset): The complete DICOM data structure.

        Returns:
            numpy.ndarray: The windowed, high bit-depth image matrix.
        """
        try:
            return apply_voi_lut(frame, dicom_data, index=0)
        except (ValueError, AttributeError):
            return frame

    def normalize_image(self, image, dicom_data):
        """
        Scales the image to 8-bit range and handles photometric interpretation.

        The function normalizes the data between 0-255, then checks the 
        Photometric Interpretation tag for potential color inversion (MONOCHROME1).

        Args:
            image (numpy.ndarray): The windowed image matrix.
            dicom_data (pydicom.dataset.FileDataset): The DICOM metadata.

        Returns:
            numpy.ndarray: 8-bit, normalized image frame.
        """
        img_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        
        if getattr(dicom_data, "PhotometricInterpretation", "") == "MONOCHROME1":
            img_8bit = cv2.bitwise_not(img_8bit)
            
        return img_8bit

    def save_frame(self, image, output_folder, dicom_name, frame_index, is_sequence):
        """
        Saves the processed frame to the disk as a JPG file in a flat structure.

        Args:
            image (numpy.ndarray): The 8-bit image to be saved.
            output_folder (str): Path to the destination folder.
            dicom_name (str): Original DICOM filename (used as prefix).
            frame_index (int): The sequence number of the frame.
            is_sequence (bool): Indicates if the file is a multi-frame sequence.
        """
        safe_prefix = dicom_name.replace(" ", "_").replace(".", "_")
        if is_sequence:
            save_name = f"{safe_prefix}_frame_{frame_index:04d}.jpg"
        else:
            save_name = f"{safe_prefix}_single.jpg"
            
        cv2.imwrite(os.path.join(output_folder, save_name), image)

    def process_single_dicom(self, dicom_path, output_root):
        """
        Processes a single DICOM file and saves all frames into a single flat directory.
        """
        try:
            filename = os.path.basename(dicom_path)
            ds = pydicom.dcmread(dicom_path)
            
            if 'PixelData' not in ds:
                self.logger.warning(f"No PixelData found in: {filename}")
                return

            if not os.path.exists(output_root):
                os.makedirs(output_root)

            pixel_data = ds.pixel_array
            is_seq = len(pixel_data.shape) == 3
            num_frames = pixel_data.shape[0] if is_seq else 1

            for i in range(num_frames):
                raw_frame = pixel_data[i] if is_seq else pixel_data
                windowed = self.apply_windowing(raw_frame, ds)
                normalized = self.normalize_image(windowed, ds)
                # Save directly to output_root without subfolders
                self.save_frame(normalized, output_root, filename, i, is_seq)
                
        except Exception as e:
            self.logger.error(f"Failed to process {dicom_path}: {str(e)}")

    def process_all_dicoms(self, input_root, output_root):
        """
        Iterates through the source directory and processes all DICOM files.

        The function lists the content of the specified input directory 
        and processes each DICOM file one by one while displaying a progress bar.

        Args:
            input_root (str): Directory containing the raw DICOM files.
            output_root (str): Root directory for the output images.

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified source directory does not exist.
        """
        if not os.path.exists(input_root):
            self.logger.error(f"Source directory not found: {input_root}")
            raise FileNotFoundError(f"Source directory not found: {input_root}")

        files = [f for f in os.listdir(input_root) if os.path.isfile(os.path.join(input_root, f))]
        self.logger.info(f"Found {len(files)} files in {input_root}")

        for filename in tqdm(files, desc="Processing DICOMs"):
            dicom_path = os.path.join(input_root, filename)
            self.process_single_dicom(dicom_path, output_root)
        
        self.logger.info("Extraction process completed.")