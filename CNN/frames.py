import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    os.makedirs(output_folder, exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

    cap.release()

    print(f"Frames extracted: {frame_count}")
    print(f"Frames per second (fps): {fps}")
    print(f"Frame dimensions: {width} x {height}")

video_path = "/Users/ajaymaheshwari/Desktop/DEV/AI_Project/Ai-project-5thsem/CNN/Movie_1.mov"
output_folder = "/Users/ajaymaheshwari/Desktop/DEV/AI_Project/Ai-project-5thsem/CNN/Images_1"
extract_frames(video_path, output_folder)