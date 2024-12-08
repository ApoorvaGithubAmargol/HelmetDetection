import cv2
import torch

# Load YOLOv5 model (make sure the model is trained on helmet detection)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can replace 'yolov5s' with a specific model if needed

# Load video file
videoFReader = cv2.VideoCapture('video.mp4')  # Make sure 'video.mp4' is in the same directory as your script
if not videoFReader.isOpened():
    print("Error: Could not open video.")
else:
    print("Video opened successfully.")

# Process video frame by frame
while True:
    ret, frame = videoFReader.read()
    if not ret:
        break  # Break the loop if no more frames are available
    
    # Detect objects in the frame
    results = model(frame)  # Detect objects using YOLOv5 model
    labels = results.names  # Get label names
    bboxes = results.xyxy[0]  # Bounding boxes of detected objects
    
    # Print detected labels for debugging
    print("Labels detected:", [results.names[int(i)] for i in bboxes[:, -1]])  # Print the label of each detected object
    
    # Draw bounding boxes and labels on the frame
    for i in range(len(bboxes)):
        label = results.names[int(bboxes[i, -1])]
        if label == 'helmet':  # Check if the label is 'helmet'
            # Draw the bounding box and label
            x1, y1, x2, y2 = map(int, bboxes[i, :4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the processed frame
    cv2.imshow('Helmet Detection', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video reader and close the window
videoFReader.release()
cv2.destroyAllWindows()