import torch
import cv2
import numpy as np
import subprocess
import os
import warnings
from ultralytics import YOLO
from segment_anything.build_sam import sam_model_registry
from segment_anything.predictor import SamPredictor
from imag_utils import resize_and_pad
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings("ignore")

def select_image_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]
    )
    return file_path

def select_directory():
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory(title="Select Output Directory")
    return dir_path

image_path = select_image_file()
if not image_path:
    print("No image selected. Exiting.")
    exit()

yolo_model = YOLO("yolo11n.pt")
sam_checkpoint = r"C:\Users\harsh\Downloads\ft\segment_anything\sam_vit_h.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to("cuda")
predictor = SamPredictor(sam)

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image_rgb)

results = yolo_model(image)
class_names = yolo_model.names

bboxes = []
class_labels = []
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = class_names.get(cls_id, f"class_{cls_id}")
        bboxes.append([x1, y1, x2, y2])
        class_labels.append(label)

yolo_output_image = image.copy()
for bbox, label in zip(bboxes, class_labels):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(yolo_output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(yolo_output_image, label, (x1, max(y1-10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

yolo_output_path = "C:/Users/harsh/Downloads/ft/yolo_detected_output.png"
cv2.imwrite(yolo_output_path, yolo_output_image)
print(f"YOLO output image saved to {yolo_output_path}")

output_dir = select_directory()
if not output_dir:
    print("No output directory selected. Exiting.")
    exit()

saved_images = []
for idx, (bbox, label) in enumerate(zip(bboxes, class_labels)):
    x1, y1, x2, y2 = bbox
    box = np.array([x1, y1, x2, y2])
    masks, _, _ = predictor.predict(box=box)

    # Create a combined mask that includes all masks to retain overlapping pixels
    combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask)

    masked_image = image.copy()
    masked_image[combined_mask == 0] = [0, 0, 0]  # Set non-masked pixels to black

    cropped_masked = masked_image[y1:y2, x1:x2].copy()
    resized_cropped = resize_and_pad(cropped_masked, size=512)

    # Display the cropped image for user decision
    window_name = f"Cropped Object {idx} - {label}"
    cv2.imshow(window_name, resized_cropped)

    # Ask the user whether to save this crop
    print(f"Do you want to save {window_name}? (y/n): ", end='')
    key = input().strip().lower()

    if key == 'y':
        safe_label = label.replace(" ", "_")
        output_path = os.path.join(output_dir, f"masked_output_{safe_label}_{x1}_{y1}.png")
        cv2.imwrite(output_path, resized_cropped)
        saved_images.append(output_path)
        print(f"Saved masked output: {output_path}")
    else:
        print(f"Skipped saving {window_name}")

    cv2.destroyWindow(window_name)

if not saved_images:
    print("No images saved, exiting.")
    exit()

# Run the batch_openpose.py script with directory change
openpose_script_path = r"C:\Users\harsh\Downloads\ft\batch_openpose.py"
openpose_dir = r"C:\Users\harsh\Downloads\ft\pifuhd"
cmd = f'cd /d "{openpose_dir}" && python "{openpose_script_path}"'

print("Running batch_openpose.py...")
result = subprocess.run(cmd, shell=True)

if result.returncode != 0:
    print("❌ batch_openpose.py failed to run. Check paths and configurations.")
else:
    print("✅ batch_openpose.py executed successfully.")

# Run the simple_test.py script
pifuhd_path = r"C:\Users\harsh\Downloads\ft\open_vscode_and_run_simpletest.py"
cmd = ['python', pifuhd_path]

print("Running simple_test.py...")
result = subprocess.run(cmd, shell=True)

if result.returncode != 0:
    print("❌ simple_test.py failed to run. Check paths and configurations.")
else:
    print("✅ simple_test.py executed successfully.")


