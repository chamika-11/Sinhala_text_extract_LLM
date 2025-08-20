"""
Sinhala Handwritten Text Recognition System

This program processes handwritten Sinhala letters using an object detection model,
sorts them into reading order, assembles text, and corrects it using a language model.

Requirements:
- pip install transformers accelerate torch pillow numpy
- Roboflow-exported model with run_model() function
- Internet connection for initial model download (then runs offline)

Author: AI Assistant
"""

import json
import os
from typing import List, Dict, Tuple, Any
import logging
from collections import defaultdict

# Standard libraries
import warnings
warnings.filterwarnings("ignore")

# Third-party libraries (install with pip)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing required library: {e}")
    print("Please install with: pip install transformers accelerate torch pillow numpy")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SinhalaTextRecognizer:
    """Main class for Sinhala handwritten text recognition and correction."""
    
    def __init__(self, word_gap_threshold: int = 40, line_gap_threshold: int = 30):
        """
        Initialize the Sinhala text recognizer.
        
        Args:
            word_gap_threshold: Minimum horizontal pixel gap to insert space between words
            line_gap_threshold: Minimum vertical pixel gap to separate lines
        """
        self.word_gap_threshold = word_gap_threshold
        self.line_gap_threshold = line_gap_threshold
        self.id_to_sinhala = self._create_class_mapping()
        self.llm_pipeline = None
        
    def _create_class_mapping(self) -> Dict[int, str]:
        """
        Create mapping from class IDs (1-39) to Sinhala Unicode characters.
        
        This mapping covers the main Sinhala vowels and consonants typically
        found in handwritten text recognition datasets.
        
        Returns:
            Dictionary mapping class ID to Sinhala character
        """
        id_to_sinhala = {
            # Vowels
            1: "අ",    # a
            2: "ආ",    # aa
            3: "ඇ",    # ae
            4: "ඈ",    # aae
            5: "ඉ",    # i
            6: "ඊ",    # ii
            7: "උ",    # u
            8: "ඌ",    # uu
            9: "ඍ",    # ri
            10: "ඎ",   # rii
            11: "ඏ",   # lu
            12: "ඐ",   # luu
            13: "එ",   # e
            14: "ඒ",   # ee
            15: "ඓ",   # ai
            16: "ඔ",   # o
            17: "ඕ",   # oo
            18: "ඖ",   # au
            
            # Consonants
            19: "ක",   # ka
            20: "ඛ",   # kha
            21: "ග",   # ga
            22: "ඝ",   # gha
            23: "ඞ",   # nga
            24: "ච",   # cha
            25: "ඡ",   # chha
            26: "ජ",   # ja
            27: "ඣ",   # jha
            28: "ඤ",   # nya
            29: "ට",   # tta
            30: "ඨ",   # ttha
            31: "ඩ",   # dda
            32: "ඪ",   # ddha
            33: "ණ",   # nna
            34: "ත",   # ta
            35: "ථ",   # tha
            36: "ද",   # da
            37: "ධ",   # dha
            38: "න",   # na
            39: "ප"    # pa
        }
        
        logger.info(f"Created class mapping for {len(id_to_sinhala)} Sinhala characters")
        return id_to_sinhala
        
    def load_language_model(self, model_name: str = "google/flan-t5-small"):
        """
        Load a language model for text correction.
        
        Uses a smaller model for better offline performance. Gemma-2B might be too large
        for some systems, so we use Flan-T5-small as default.
        
        Args:
            model_name: HuggingFace model name
        """
        try:
            logger.info(f"Loading language model: {model_name}")
            
            # Check if we have GPU available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load the pipeline for text generation
            self.llm_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                max_length=512
            )
            
            logger.info("Language model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load language model: {e}")
            logger.info("Continuing without language model correction...")
            self.llm_pipeline = None
    
    def sort_detections_by_reading_order(self, detections: List[Dict]) -> List[Dict]:
        """
        Sort detections into proper Sinhala reading order (left-to-right, top-to-bottom).
        
        Args:
            detections: List of detection dictionaries with x, y coordinates
            
        Returns:
            Sorted list of detections in reading order
        """
        if not detections:
            return []
            
        logger.info(f"Sorting {len(detections)} detections into reading order")
        
        # Group detections by line based on Y-coordinate
        lines = defaultdict(list)
        
        # Sort all detections by Y coordinate first
        detections_by_y = sorted(detections, key=lambda d: d['y'])
        
        # Group detections into lines
        current_line_y = detections_by_y[0]['y']
        current_line_id = 0
        
        for detection in detections_by_y:
            # If Y difference is greater than threshold, start new line
            if abs(detection['y'] - current_line_y) > self.line_gap_threshold:
                current_line_id += 1
                current_line_y = detection['y']
            
            lines[current_line_id].append(detection)
        
        # Sort detections within each line by X coordinate (left to right)
        sorted_detections = []
        for line_id in sorted(lines.keys()):
            line_detections = sorted(lines[line_id], key=lambda d: d['x'])
            sorted_detections.extend(line_detections)
        
        logger.info(f"Organized detections into {len(lines)} lines")
        return sorted_detections
    
    def assemble_text_from_detections(self, sorted_detections: List[Dict]) -> str:
        """
        Convert sorted detections to Sinhala text with appropriate spacing.
        
        Args:
            sorted_detections: Detections sorted in reading order
            
        Returns:
            Assembled Sinhala text string
        """
        if not sorted_detections:
            return ""
            
        logger.info("Assembling text from detections")
        
        text_parts = []
        prev_detection = None
        
        for detection in sorted_detections:
            class_id = detection['class']
            
            # Convert class ID to Sinhala character
            if class_id in self.id_to_sinhala:
                char = self.id_to_sinhala[class_id]
                
                # Add space if horizontal gap is large enough
                if prev_detection is not None:
                    horizontal_gap = detection['x'] - prev_detection['x']
                    vertical_gap = abs(detection['y'] - prev_detection['y'])
                    
                    # Add space for word separation or new line
                    if horizontal_gap > self.word_gap_threshold:
                        text_parts.append(' ')
                    elif vertical_gap > self.line_gap_threshold:
                        text_parts.append('\n')
                
                text_parts.append(char)
                prev_detection = detection
            else:
                logger.warning(f"Unknown class ID: {class_id}")
        
        assembled_text = ''.join(text_parts)
        logger.info(f"Assembled text with {len(assembled_text)} characters")
        return assembled_text
    
    def correct_text_with_llm(self, raw_text: str) -> str:
        """
        Use language model to correct Sinhala text spelling and joiners.
        
        Args:
            raw_text: Raw OCR output text
            
        Returns:
            Corrected text or original text if model unavailable
        """
        if not self.llm_pipeline or not raw_text.strip():
            logger.warning("Language model not available or empty text")
            return raw_text
        
        try:
            logger.info("Correcting text with language model")
            
            # Create prompt for text correction
            prompt = f"Correct the Sinhala spelling and joiners in this text: {raw_text}"
            
            # Generate corrected text
            result = self.llm_pipeline(
                prompt,
                max_length=len(raw_text) + 100,
                num_return_sequences=1,
                do_sample=False,
                temperature=0.1
            )
            
            corrected_text = result[0]['generated_text']
            logger.info("Text correction completed")
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error in text correction: {e}")
            return raw_text
    
    def save_results(self, raw_text: str, corrected_text: str, 
                    detections: List[Dict], output_dir: str = "."):
        """
        Save recognition results to files.
        
        Args:
            raw_text: Raw OCR output
            corrected_text: LLM-corrected text
            detections: Original detections with coordinates
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw OCR output
        raw_file = os.path.join(output_dir, "raw_output.txt")
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        logger.info(f"Saved raw output to: {raw_file}")
        
        # Save corrected text
        corrected_file = os.path.join(output_dir, "corrected_output.txt")
        with open(corrected_file, 'w', encoding='utf-8') as f:
            f.write(corrected_text)
        logger.info(f"Saved corrected output to: {corrected_file}")
        
        # Save detailed results as JSON
        results_data = {
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "detection_count": len(detections),
            "detections": detections,
            "settings": {
                "word_gap_threshold": self.word_gap_threshold,
                "line_gap_threshold": self.line_gap_threshold
            }
        }
        
        json_file = os.path.join(output_dir, "recognition_results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved detailed results to: {json_file}")
    
    def process_image(self, image_path: str, output_dir: str = ".") -> Tuple[str, str]:
        """
        Complete pipeline to process a handwritten Sinhala image.
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            
        Returns:
            Tuple of (raw_text, corrected_text)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Step 1: Run object detection (assuming run_model function exists)
        try:
            detections = run_model(image_path)
            logger.info(f"Detected {len(detections)} letters")
        except NameError:
            logger.error("run_model() function not found. Please implement your model inference.")
            # Create dummy detections for testing
            detections = [
                {"x": 100, "y": 50, "class": 1, "confidence": 0.95},
                {"x": 140, "y": 52, "class": 19, "confidence": 0.93},
                {"x": 180, "y": 48, "class": 34, "confidence": 0.91},
            ]
            logger.warning("Using dummy detections for demonstration")
        
        # Step 2: Sort detections into reading order
        sorted_detections = self.sort_detections_by_reading_order(detections)
        
        # Step 3: Assemble text from detections
        raw_text = self.assemble_text_from_detections(sorted_detections)
        
        # Step 4: Correct text with language model
        corrected_text = self.correct_text_with_llm(raw_text)
        
        # Step 5: Save results
        self.save_results(raw_text, corrected_text, detections, output_dir)
        
        logger.info("Processing completed successfully")
        return raw_text, corrected_text


def run_model(image_path: str) -> List[Dict]:
    """
    Placeholder function for Roboflow model inference.
    
    Replace this with your actual model loading and inference code.
    This should return detections in the specified format.
    
    Args:
        image_path: Path to input image
        
    Returns:
        List of detection dictionaries
    """
    # TODO: Replace with actual model inference
    # Example implementation:
    # model = torch.load('your_model.pt')
    # image = Image.open(image_path)
    # detections = model.predict(image)
    # return format_detections(detections)
    
    logger.warning("run_model() not implemented. Using dummy data.")
    
    # Dummy detections for testing - replace with your model
    dummy_detections = [
        {"x": 50, "y": 100, "class": 1, "confidence": 0.95},   # අ
        {"x": 90, "y": 102, "class": 19, "confidence": 0.93},  # ක
        {"x": 140, "y": 98, "class": 34, "confidence": 0.91},  # ත
        {"x": 200, "y": 101, "class": 38, "confidence": 0.89}, # න
        {"x": 50, "y": 150, "class": 13, "confidence": 0.87},  # එ (new line)
        {"x": 90, "y": 152, "class": 19, "confidence": 0.85},  # ක
    ]
    
    return dummy_detections


def main():
    """Main function to run the Sinhala text recognition system."""
    
    # Configuration
    IMAGE_PATH = "handwritten_sinhala.jpg"  # Replace with your image path
    OUTPUT_DIR = "output"
    WORD_GAP_THRESHOLD = 40  # Pixels
    LINE_GAP_THRESHOLD = 30  # Pixels
    
    # Initialize recognizer
    recognizer = SinhalaTextRecognizer(
        word_gap_threshold=WORD_GAP_THRESHOLD,
        line_gap_threshold=LINE_GAP_THRESHOLD
    )
    
    # Load language model for correction
    recognizer.load_language_model("google/flan-t5-small")
    
    # Process image
    try:
        raw_text, corrected_text = recognizer.process_image(IMAGE_PATH, OUTPUT_DIR)
        
        # Display results
        print("\n" + "="*50)
        print("SINHALA TEXT RECOGNITION RESULTS")
        print("="*50)
        print(f"Raw OCR Output:\n{raw_text}")
        print("-"*30)
        print(f"Corrected Text:\n{corrected_text}")
        print("="*50)
        print(f"Files saved to: {OUTPUT_DIR}/")
        
    except FileNotFoundError:
        logger.error(f"Image file not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH with your actual image file.")
    except Exception as e:
        logger.error(f"Error processing image: {e}")


if __name__ == "__main__":
    main()