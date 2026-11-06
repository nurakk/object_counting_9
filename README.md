# Experiment 9: Object Counting Using YOLOv8

## 1. Project Overview

This project demonstrates object detection and counting using the YOLOv8 (You Only Look Once) deep learning algorithm.

The program uses the YOLOv8 Nano model to identify objects in an input image, classify them into different categories and calculate the number of detected objects in each category.

The experiment uses the same sample bus image used in Experiment 8.

## 2. Objectives

- Understand object detection using YOLOv8.
- Detect multiple objects in a single image.
- Identify and classify detected objects.
- Calculate class-wise object counts.
- Display bounding boxes and confidence scores.
- Visualize object counts using a bar graph.

## 3. Algorithm Used

**YOLOv8 – You Only Look Once**

YOLOv8 is a single-stage deep-learning-based object detection model. It processes an image and predicts object bounding boxes, class labels and confidence scores.

Python's `Counter` function is used to calculate the number of detected objects in each class.

## 4. Technologies and Libraries

- Programming Language: Python
- Deep Learning Model: YOLOv8 Nano
- Framework: Ultralytics
- Image Processing: OpenCV
- Data Visualization: Matplotlib
- Object Counting: Python Collections
- Environment: Jupyter Notebook

## 5. Input Image

The experiment uses the official Ultralytics sample image:

https://ultralytics.com/images/bus.jpg

The image contains a bus and people, making it suitable for demonstrating object detection and counting.

## 6. Installation

Install the required libraries:

```bash
pip install ultralytics opencv-python matplotlib
```

## 7. Execution Procedure

1. Clone or download this repository.
2. Open `Experiment_9_Object_Counting.ipynb` in Jupyter Notebook.
3. Install the required Python libraries.
4. Ensure that `bus.jpg` is available in the working directory.
5. Execute the notebook cells sequentially.
6. Observe the detected objects and their class-wise counts.

## 8. Expected Output

The program generates:

- Original input image.
- Object detection results with bounding boxes.
- Detected object names and confidence scores.
- Class-wise object counts.
- Total number of detected objects.
- Bar graph showing the number of objects in each class.

**Note:** Actual detection counts may vary depending on the model version and confidence threshold.

## 9. Project Files

- `Experiment_9_Object_Counting.ipynb` – Complete Jupyter Notebook program.
- `bus.jpg` – Sample input image.
- `requirements.txt` – Required Python libraries.

## 10. Applications

- Autonomous mobile robots
- Traffic monitoring and vehicle counting
- Industrial object counting
- Smart surveillance systems
- Automated inventory management
- Robotics vision applications

## 11. Author

**Arun Kumar K**  
Department of Robotics and Automation

## 12. References

- Ultralytics Documentation: https://docs.ultralytics.com/
- Ultralytics GitHub: https://github.com/ultralytics/ultralytics
