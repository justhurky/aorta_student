"""
Module for running inference using trained YOLO weights.
Handles visual result generation, calculates horizontal distances in mm 
using DICOM metadata, and exports results to Excel.
Visualizes measurements with a strictly HORIZONTAL yellow line.
"""

import os
import cv2
import pydicom
import pandas as pd
from ultralytics import YOLO

from fpdf import FPDF

class ModelPredictor:
    """
    A class to manage professional medical inference. Performs spatial analysis,
    pixel-to-mm conversion via DICOM metadata, and visual annotation.
    """

    def __init__(self, logger):
        """
        Initializes the ModelPredictor with a centralized logger.

        Args:
            logger (logging.Logger): Logger instance for status reporting.
        """
        self.logger = logger
        self.inference_results = [] # To store data for PDF export

    def _get_pixel_spacing(self, dicom_path):
        """
        Extracts the pixel spacing (mm/pixel) from a DICOM file.
        Checks for PixelSpacing, then ImagerPixelSpacing as a fallback.
        """
        try:
            if dicom_path and os.path.exists(dicom_path):
                self.logger.info(f"Reading DICOM file for calibration: {dicom_path}")
                ds = pydicom.dcmread(dicom_path)
                
                # Priority 1: PixelSpacing (0028,0030)
                if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
                    spacing = float(ds.PixelSpacing[1])
                    self.logger.info(f"Using PixelSpacing: {spacing} mm/px")
                    return spacing
                
                # Priority 2: ImagerPixelSpacing (0018,1164)
                if hasattr(ds, "ImagerPixelSpacing") and ds.ImagerPixelSpacing:
                    spacing = float(ds.ImagerPixelSpacing[1])
                    self.logger.info(f"Using ImagerPixelSpacing: {spacing} mm/px")
                    return spacing

                self.logger.warning("No spacing metadata found in DICOM. Using 1.0 ratio.")
        except Exception as e:
            self.logger.warning(f"Error reading DICOM metadata ({e}). Using 1.0 ratio.")
        return 1.0

    def _draw_measurement(self, image, box1_coords, box2_coords, dist_mm, output_path):
        """
        Draws a strictly horizontal yellow line between the X coordinates 
        of two points and labels the distance in mm.
        """
        # Center coordinates
        x1, y1 = int(box1_coords[0]), int(box1_coords[1])
        x2, y2 = int(box2_coords[0]), int(box2_coords[1])

        # Calculate a common Y level (the average of the two centers)
        # This ensures the line is perfectly horizontal regardless of vertical offset
        y_level = int((y1 + y2) / 2)

        # Draw the horizontal yellow line 
        cv2.line(image, (x1, y_level), (x2, y_level), (0, 255, 255), 3)
        
        # Draw vertical 'caps' at the ends of the line (technical drawing style)
        cv2.line(image, (x1, y_level - 15), (x1, y_level + 15), (0, 255, 255), 2)
        cv2.line(image, (x2, y_level - 15), (x2, y_level + 15), (0, 255, 255), 2)

        # Add text label with mm value above the horizontal line
        label = f"{dist_mm:.2f} mm"
        text_x = min(x1, x2)
        text_y = y_level - 20
        cv2.putText(image, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        
        # Save the annotated visual output
        cv2.imwrite(output_path, image)

    def run_prediction(self, model_path, source_input, output_dir, dicom_ref=None, conf=0.45):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            spacing = self._get_pixel_spacing(dicom_ref)
            model = YOLO(model_path)
            results = model.predict(source=source_input, project=output_dir, name="measurements", save=True, conf=conf, exist_ok=True)
            self._process_and_save(results, spacing, output_dir, model_path)
            self._export_to_pdf(output_dir)
        except Exception as e:
            self.logger.error(f"Batch inference failed: {str(e)}")
            raise

    def run_single_prediction(self, model_path, file_path, output_dir, dicom_ref=None, conf=0.45):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            spacing = self._get_pixel_spacing(dicom_ref)
            model = YOLO(model_path)
            results = model.predict(source=file_path, project=output_dir, name="single_run", save=True, conf=conf, exist_ok=True)
            self._process_and_save(results, spacing, output_dir, model_path)
            self._export_to_pdf(output_dir)
        except Exception as e:
            self.logger.error(f"Single prediction failed: {str(e)}")
            raise

    def run_video_prediction(self, model_path, video_path, output_dir, dicom_ref=None, conf=0.45):
        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            spacing = self._get_pixel_spacing(dicom_ref)
            model = YOLO(model_path)
            results = model.predict(source=video_path, project=output_dir, name="video_analysis", save=True, conf=conf, exist_ok=True, device='cpu')
            self._process_and_save(results, spacing, output_dir, model_path)
            self._export_to_pdf(output_dir)
        except Exception as e:
            self.logger.error(f"Video analysis failed: {str(e)}")
            raise

    def _process_and_save(self, results, spacing, output_dir, model_name):
        """
        Internal logic to calculate strictly horizontal distance and log data.
        """
        for result in results:
            img_name = os.path.basename(result.path)
            boxes = result.boxes

            if len(boxes) < 2:
                self.logger.warning(f"Detection count < 2 for {img_name}. Skipping distance measurement.")
                continue

            # Extract center X and Y
            b1_data = boxes[0].xywh[0].tolist() 
            b2_data = boxes[1].xywh[0].tolist()

            # Horizontal distance calculation
            dist_px = abs(b1_data[0] - b2_data[0])
            dist_mm = dist_px * spacing

            print(f">>> HORIZONTAL MEASUREMENT [{img_name}]: {dist_px:.2f} px | {dist_mm:.2f} mm")

            # Draw visual annotation
            annotated_img = result.orig_img.copy()
            save_path = os.path.join(output_dir, f"measured_{img_name}")
            self._draw_measurement(
                annotated_img, 
                (b1_data[0], b1_data[1]), 
                (b2_data[0], b2_data[1]), 
                dist_mm, 
                save_path
            )

            # Store for PDF
            self.inference_results.append({
                "Filename": img_name,
                "Model": os.path.basename(model_name),
                "Pixel_Dist_H": round(dist_px, 2),
                "MM_Dist_H": round(dist_mm, 2),
                "Confidence_Avg": round((float(boxes[0].conf) + float(boxes[1].conf)) / 2, 4)
            })

    def _export_to_pdf(self, output_dir):
        if not self.inference_results:
            return
        
        pdf_file = os.path.join(output_dir, "inference_report.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Medical Imaging: YOLO Measurement Report", ln=True, align='C')
        pdf.ln(10)
        
        # Table Header
        pdf.set_font("Helvetica", "B", 10)
        col_width = pdf.epw / 5
        pdf.cell(col_width, 10, "Filename", border=1)
        pdf.cell(col_width, 10, "Model", border=1)
        pdf.cell(col_width, 10, "Dist (px)", border=1)
        pdf.cell(col_width, 10, "Dist (mm)", border=1)
        pdf.cell(col_width, 10, "Conf (avg)", border=1)
        pdf.ln()
        
        # Table Content
        pdf.set_font("Helvetica", "", 10)
        for row in self.inference_results:
            pdf.cell(col_width, 10, str(row["Filename"]), border=1)
            pdf.cell(col_width, 10, str(row["Model"]), border=1)
            pdf.cell(col_width, 10, str(row["Pixel_Dist_H"]), border=1)
            pdf.cell(col_width, 10, str(row["MM_Dist_H"]), border=1)
            pdf.cell(col_width, 10, str(row["Confidence_Avg"]), border=1)
            pdf.ln()
            
        pdf.output(pdf_file)
        self.logger.info(f"PDF report generated: {pdf_file}")