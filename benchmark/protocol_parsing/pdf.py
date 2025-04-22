"""
This file contains the code for parsing the protocol from a PDF file.

Using pymupdf4llm for enhanced PDF parsing with better table extraction and markdown formatting.
"""

import os
import io
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pymupdf4llm
import fitz  # PyMuPDF
from PIL import Image
import numpy as np


@dataclass
class Figure:
    """Class for storing figure data extracted from PDF."""
    image: bytes
    page_number: int
    bbox: Tuple[float, float, float, float]
    name: str
    width: int
    height: int
    size_kb: float


def extract_pdf_text(pdf_path):
    """
    Extract the text from the PDF file.
    """
    return pymupdf4llm.to_markdown(pdf_path)


def extract_markdown_pages(pdf_path) -> List[str]:
    """
    Extract text content from each page of the PDF and format as markdown.
    
    Using pymupdf4llm to properly handle tables and text formatting.
    
    Returns:
        List of strings, each containing markdown-formatted content of a page
    """
    # Use page_chunks to get a list of page dictionaries
    page_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)
    
    # Extract just the markdown text from each page
    markdown_pages = [page["text"] for page in page_data]
    
    return markdown_pages


def is_blank_or_uniform_image(pil_img, threshold=0.95):
    """
    Check if image is blank, nearly uniform in color, or likely invisible.
    
    Args:
        pil_img: PIL Image object
        threshold: Threshold for uniformity (0-1)
        
    Returns:
        True if image is likely blank/uniform, False otherwise
    """
    # Convert to grayscale for simpler analysis
    gray_img = pil_img.convert('L')
    
    # Check if image has transparency and is mostly transparent
    if 'A' in pil_img.getbands():
        transparent = np.array(pil_img.getchannel('A'))
        if np.mean(transparent) < 10:  # Very transparent on average
            return True
    
    # Convert to numpy array for histogram analysis
    img_array = np.array(gray_img)
    
    # Check if image is nearly blank (mostly white or black)
    mean_val = np.mean(img_array)
    if mean_val > 245 or mean_val < 10:
        return True
    
    # Check image variance - low variance suggests uniform color
    std_dev = np.std(img_array)
    if std_dev < 20:
        return True
    
    # Check histogram to see if a single color dominates the image
    hist = gray_img.histogram()
    total_pixels = pil_img.width * pil_img.height
    
    # Calculate if any single intensity value accounts for most pixels
    for count in hist:
        if count / total_pixels > threshold:
            return True
    
    return False


def extract_figures(
    pdf_path, 
    output_dir: Optional[str] = None,
    min_width: int = 100,  # Minimum width in pixels
    min_height: int = 100,  # Minimum height in pixels
    min_size_kb: float = 5.0,  # Minimum file size in KB
    max_aspect_ratio: float = 10.0,  # Maximum width/height or height/width ratio
    filter_invisible: bool = True  # Filter out images that appear to be invisible/decorative
) -> List[Figure]:
    """
    Extract meaningful figures from the PDF with page number metadata using PyMuPDF.
    
    Filters out small, low-quality, or irrelevant images based on size and dimension criteria.
    Also filters out invisible or decorative elements that aren't actually visible content.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save extracted images
        min_width: Minimum width in pixels to consider an image meaningful
        min_height: Minimum height in pixels to consider an image meaningful
        min_size_kb: Minimum size in KB to consider an image meaningful
        max_aspect_ratio: Maximum allowed aspect ratio (width/height or height/width)
        filter_invisible: Whether to filter out images that appear to be invisible/decorative
    
    Returns:
        List of Figure objects containing image data and metadata
    """
    # Open the document with PyMuPDF
    doc = fitz.open(pdf_path)
    figures = []
    
    # Process each page
    for page_num, page in enumerate(doc):
        # Get page dimensions for later bounds checking
        page_rect = page.rect
        
        # Get all images on the page
        image_list = page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]  # Image reference number
            
            # Extract image metadata
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Get image properties
                img = Image.open(io.BytesIO(image_bytes))
                width, height = img.size
                size_kb = len(image_bytes) / 1024  # Size in KB
                
                # Calculate aspect ratio
                aspect_ratio = max(width / max(height, 1), height / max(width, 1))
                
                # Filter out small or oddly proportioned images
                if (width < min_width or 
                    height < min_height or 
                    size_kb < min_size_kb or 
                    aspect_ratio > max_aspect_ratio):
                    continue  # Skip this image
                
                # Filter out invisible or decorative images
                if filter_invisible and is_blank_or_uniform_image(img):
                    continue  # Skip images that appear blank or uniform
                
                # Try to get the position of the image on the page
                try:
                    rect = page.get_image_bbox(img_info)
                    
                    # Skip if image is outside page bounds or has zero/negative area
                    if (rect.x0 >= rect.x1 or rect.y0 >= rect.y1 or
                        rect.x1 <= 0 or rect.y1 <= 0 or
                        rect.x0 >= page_rect.width or rect.y0 >= page_rect.height):
                        continue
                        
                except Exception:
                    # If we can't get the position, use a default
                    rect = fitz.Rect(0, 0, width, height)
                
                # Create a unique name for the figure
                name = f"image_{page_num+1}_{img_idx+1}"
                
                # Create the Figure object
                figure = Figure(
                    image=image_bytes,
                    page_number=page_num + 1,  # 1-indexed page number
                    bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                    name=name,
                    width=width,
                    height=height,
                    size_kb=size_kb
                )
                figures.append(figure)
                
                # Save the image to disk if output_dir is specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    file_path = os.path.join(output_dir, f"{name}.png")
                    img.save(file_path)
            
            except Exception as e:
                # Skip images that cause extraction errors
                print(f"Skipping image {img_idx} on page {page_num+1}: {e}")
    
    return figures


def save_extracted_figures(figures: List[Figure], output_dir: str) -> Dict[str, str]:
    """
    Save extracted figures to disk and return mapping of names to file paths.
    
    Args:
        figures: List of Figure objects
        output_dir: Directory to save images
        
    Returns:
        Dictionary mapping figure names to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = {}
    
    for figure in figures:
        # Save image to file
        image = Image.open(io.BytesIO(figure.image))
        file_path = os.path.join(output_dir, f"{figure.name}.png")
        image.save(file_path)
        
        figure_paths[figure.name] = file_path
    
    return figure_paths