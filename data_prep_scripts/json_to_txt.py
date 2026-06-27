import os
import json

class JsonToTxtConverter:
    """
    A class to convert VoTT JSON annotations to normalized YOLO TXT format.
    """

    def __init__(self, logger):
        """
        Initializes the converter with a logger and a mapping for tags.

        Args:
            logger (logging.Logger): The logger instance for status updates and errors.
        """
        self.logger = logger
        # Mapping VoTT tags to integer class IDs
        self.tag_map = {
            "rec1": 0,
            "rec2": 1
        }

    def _extract_data(self, json_content):
        """
        Extracts and normalizes region data from VoTT JSON to YOLO format.

        This method calculates the normalized center coordinates, width, and height 
        based on the image size provided in the JSON asset metadata.

        Args:
            json_content (dict): The loaded JSON data from a VoTT export file.

        Returns:
            list: A list of strings, each representing a bounding box in 
                  'class_id x_center y_center width height' format.
        """
        lines = []
        
        # Extract image dimensions for normalization
        asset = json_content.get("asset", {})
        size = asset.get("size", {})
        img_w = size.get("width", 1024)
        img_h = size.get("height", 1024)

        for region in json_content.get("regions", []):
            tags = region.get("tags", [])
            if not tags:
                continue
            
            # Map the first tag to a class ID
            class_id = self.tag_map.get(tags[0], 0)

            # Get pixel-based bounding box coordinates
            bbox = region.get("boundingBox", {})
            left = bbox.get("left", 0)
            top = bbox.get("top", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)

            # Calculate YOLO format: normalized center coordinates
            x_center = (left + (width / 2)) / img_w
            y_center = (top + (height / 2)) / img_h
            
            # Calculate normalized width and height
            w_norm = width / img_w
            h_norm = height / img_h

            # Format the line with high precision for better model training
            line = f"{class_id} {x_center:.18f} {y_center:.18f} {w_norm:.18f} {h_norm:.18f}"
            lines.append(line)
            
        return lines

    def convert_single_file(self, json_path, output_folder):
        """
        Converts a single JSON file to its TXT counterpart.
        Uses the original image name stored in the JSON asset for the TXT filename.

        Args:
            json_path (str): Full path to the source JSON file.
            output_folder (str): Directory where the TXT file will be saved.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            lines = self._extract_data(data)
            
            # Get original image name from asset metadata
            asset_name = data.get("asset", {}).get("name", "")
            if asset_name:
                # Replace image extension with .txt
                filename = os.path.splitext(asset_name)[0] + ".txt"
            else:
                # Fallback to JSON filename if asset name is missing
                filename = os.path.basename(json_path).replace(".json", ".txt")

            output_path = os.path.join(output_folder, filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            
            self.logger.info(f"Converted: {os.path.basename(json_path)} -> {filename}")
            
        except Exception as e:
            self.logger.error(f"Error converting file {json_path}: {str(e)}")

    def process_all(self, input_root, output_root):
        """
        Processes all JSON files in the input directory and saves them to the output directory.

        Args:
            input_root (str): Source directory containing VoTT JSON files.
            output_root (str): Destination directory for the generated TXT files.
        """
        if not os.path.exists(input_root):
            self.logger.error(f"Input directory not found: {input_root}")
            return

        if not os.path.exists(output_root):
            os.makedirs(output_root)

        # Filter for JSON files only
        json_files = [f for f in os.listdir(input_root) if f.endswith('.json')]
        self.logger.info(f"Found {len(json_files)} JSON files to convert.")

        for filename in json_files:
            self.convert_single_file(os.path.join(input_root, filename), output_root)

        self.logger.info("Conversion process completed.")