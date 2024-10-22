import cv2
import os
from ultralytics import YOLO

# Load the trained model
model = YOLO(r'C:\Users\Bekhzod\OneDrive\Ishchi stol\dog_head\runs\detect\train\weights\best.pt')

# Path to the video file
video_path = r"C:\Users\Bekhzod\Downloads\2.mp4"  # Replace with the actual path to your video

# Create a directory to save the output video if it doesn't exist
output_folder = r"C:\Users\Bekhzod\OneDrive\Ishchi stol\dog_head\output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the output video path
output_video_path = os.path.join(output_folder, "output_video.mp4")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the video's width, height, and frames per second (FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the frame (optional)
    cv2.imshow('Dog Head Detection', annotated_frame)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, writer, and close windows
cap.release()
out.release()  # Save the output video
cv2.destroyAllWindows()

print(f"Output video saved at: {output_video_path}")
