import xml.etree.ElementTree as ET
import os
from PIL import Image

def convert_xml_to_yolo(image_folder, annotation_folder, label_folder):
    # Ensure the label folder exists
    os.makedirs(label_folder, exist_ok=True)

    # Loop through the XML files in the annotation folder
    for xml_name in os.listdir(annotation_folder):
        if xml_name.endswith('.xml'):
            # Corresponding image file
            image_file = os.path.join(image_folder, xml_name.replace('.xml', '.png'))
            xml_file = os.path.join(annotation_folder, xml_name)

            # Check if the image file exists for the XML
            if os.path.exists(image_file):
                # Process the XML and create YOLO annotation
                convert_image_and_annotation(image_file, xml_file, label_folder)

def convert_image_and_annotation(image_path, xml_path, label_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image = Image.open(image_path)
    width, height = image.size

    yolo_format = []

    for obj in root.findall('object'):
        class_name = obj.find('name').text
        # Assuming class_name is 'license_plate' and it maps to class ID 0
        class_id = 0
        
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Normalize bounding box coordinates
        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        # Append the normalized data for YOLO format
        yolo_format.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

    # Save the annotations to the corresponding .txt file
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(label_path, f"{base_filename}.txt"), 'w') as f:
        for line in yolo_format:
            f.write(line + '\n')


image_folder = "E:\\License Plate Number Detection\\archive\\images\\test"  # Path to your images folder
annotation_folder = "E:\\License Plate Number Detection\\archive\\annotations\\test"  # Path to your XML folder
label_folder = "E:\\License Plate Number Detection\\archive\\label\\test"  # Path where YOLO annotation files will be saved
convert_xml_to_yolo(image_folder, annotation_folder, label_folder)

image_folder = "E:\\License Plate Number Detection\\archive\\images\\train" 
annotation_folder = "E:\\License Plate Number Detection\\archive\\annotations\\train"  
label_folder = "E:\\License Plate Number Detection\\archive\\label\\train" 
convert_xml_to_yolo(image_folder, annotation_folder, label_folder)

image_folder = "E:\\License Plate Number Detection\\archive\\images\\validation"  
annotation_folder = "E:\\License Plate Number Detection\\archive\\annotations\\validation" 
label_folder = "E:\\License Plate Number Detection\\archive\\label\\validation" 
convert_xml_to_yolo(image_folder, annotation_folder, label_folder)