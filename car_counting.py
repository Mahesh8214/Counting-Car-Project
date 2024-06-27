import numpy as np
import cv2
import cvzone
import math
from ultralytics import YOLO
from sort import *

# Initialize video capture from a file
cap = cv2.VideoCapture("Videos/cars.mp4")

# Initialize YOLO model for object detection
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Define class names for detected objects
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Load the mask image for region of interest
mask = cv2.imread("images/mask-1280-720.png")

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Define counting line limits and initialize count list
limits = [400, 297, 673, 297]
totalCount = []

while True:
    # Read a frame from the video
    success, img = cap.read()
    if not success:
        break  # Exit the loop if video cannot be read or end of video is reached
    
    # Apply bitwise operation to region of interest defined by the mask
    imgRegion = cv2.bitwise_and(img, mask)

    # Overlay additional graphics on the frame
    imgGraphics = cv2.imread("images/graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    # Perform object detection using YOLO model
    results = model(imgRegion, stream=True)

    # Initialize array for storing detections
    detections = np.empty((0, 5))

    # Process each detected object
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate width and height of the bounding box
            w, h = x2 - x1, y2 - y1

            # Calculate confidence and determine class name
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter objects of interest and confidence threshold
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Update tracker with detected objects
    resultsTracker = tracker.update(detections)

    # Draw counting line on the frame
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    # Process each tracked object
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw rectangle around tracked object
        cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=9, rt=2, colorR=(255, 0, 255))
        
        # Display object ID near the object
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)
        
        # Draw circle at the center of the object
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        # Check if object crosses the counting line
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if id not in totalCount:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    # Display total count of objects
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

    # Display the processed frame
    cv2.imshow("Car Counting", img)

    # Wait for key press and check if 'q' is pressed to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('The total Car Passed are : ', len(totalCount))
# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
