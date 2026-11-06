#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ============================================================
# EXPERIMENT 9: OBJECT COUNTING USING YOLOv8
# ============================================================

# Step 1: Import required libraries
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# Step 2: Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Step 3: Load input image
image_path = "bus.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("bus.jpg not found. Run Experiment 8 first.")

# Step 4: Perform object detection
results = model.predict(
    source=image_path,
    conf=0.25
)

# Step 5: Count detected objects
detected_classes = []

for box in results[0].boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    detected_classes.append(class_name)

# Count objects in each class
object_counts = Counter(detected_classes)

# Step 6: Display object counts
print("OBJECT COUNTING RESULTS")
print("------------------------------")

total_objects = 0

for object_name, count in object_counts.items():
    print(object_name, ":", count)
    total_objects += count

print("------------------------------")
print("Total detected objects:", total_objects)

# Step 7: Display detected image with bounding boxes
result_image = results[0].plot()
result_image = cv2.cvtColor(
    result_image,
    cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(12, 8))
plt.imshow(result_image)
plt.title("YOLOv8 Object Detection and Counting")
plt.axis("off")
plt.show()

# Step 8: Display class-wise count graph
plt.figure(figsize=(9, 5))

plt.bar(
    object_counts.keys(),
    object_counts.values()
)

plt.xlabel("Object Class")
plt.ylabel("Number of Objects")
plt.title("Class-wise Object Counting Using YOLOv8")
plt.grid(axis="y", alpha=0.3)
plt.show()

print("Experiment completed successfully!")


# In[ ]:




