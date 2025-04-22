"""
This file contains the code for parsing the protocol from a PDF file.

Using the `pymupdf` library to parse the PDF file.
"""

import os
import pdb
import fitz
def extract_pdf_text(pdf_path):
    """
    Extract the text from the PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text