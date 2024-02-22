# pdf_logic.py

import fitz  # PyMuPDF
import io
from PIL import Image

def get_n_text_sections(pdf_path, n_pages):
    pdf_document = fitz.open(pdf_path)
    text_sections = []
    insert_points = range(0, len(pdf_document), n_pages)
    
    for page_number in insert_points:
        text_section = ''
        for i in range(n_pages):
            if page_number + i < len(pdf_document):
                page_text = pdf_document[page_number + i].get_text()
                text_section += page_text
        text_sections.append(text_section)
    
    pdf_document.close()
    return text_sections

def insert_images_and_save(pdf_path, pil_images, n_pages, output_path):
    pdf_document = fitz.open(pdf_path)
    insert_points = range(0, len(pdf_document), n_pages)

    added_pages_count = 0  # Keep track of the number of pages we've added
    for i, (insert_point, pil_image) in enumerate(zip(insert_points, pil_images)):
        # Calculate the position for a new page to insert the PIL image
        # It ensures that the page is inserted within the range of existing pages.
        new_page_index = insert_point + n_pages + added_pages_count

        # Convert PIL image to a bytes object
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)  # Move to the beginning of the BytesIO stream

        # Add a new page to the PDF
        # Ensure we're not trying to insert beyond the current page count
        if new_page_index > len(pdf_document):  # Make sure the index is within the valid range
            new_page_index = len(pdf_document)  # If it's not, adjust it to be at the end
        new_page = pdf_document.new_page(pno=new_page_index - 1)  # PyMuPDF is zero-indexed

        # Calculate the position for centering the image on the page
        page_width = new_page.rect.width
        page_height = new_page.rect.height
        image_width, image_height = pil_image.size
        left = (page_width - image_width) / 2  # Horizontal centering
        top = (page_height - image_height) / 2  # Vertical centering
        img_rect = fitz.Rect(left, top, left + image_width, top + image_height)

        # Insert the image onto the new page
        new_page.insert_image(img_rect, stream=img_bytes)

        added_pages_count += 1  # Increment the count of added pages

    # Save the modified PDF
    pdf_document.save(output_path)
    pdf_document.close()