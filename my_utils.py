import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT_PICTURE = "./pic"
ROOT_VIDEO = "./vdo"


def video_to_frame(
    video_path: str, fps: int = 1, start_time: int = 0, end_time: int = None
):
    frames_dir = os.path.join(ROOT_PICTURE, "frames")

    if not os.path.exists(ROOT_PICTURE):
        os.makedirs(ROOT_PICTURE)
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    if os.path.exists(video_path):
        video = cv.VideoCapture(video_path)

        if not video.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return

        video_fps = video.get(cv.CAP_PROP_FPS)
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / video_fps

        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps) if end_time else total_frames

        if start_frame >= total_frames or start_frame >= end_frame:
            print("Error: Invalid start or end time.")
            video.release()
            return

        video.set(cv.CAP_PROP_POS_FRAMES, start_frame)

        frame_count = 0
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = video.read()
            if not ret:
                break

            if (current_frame - start_frame) % int(video_fps / fps) == 0:
                frame_filename = os.path.join(
                    frames_dir, f"frame_{frame_count:04d}.jpg"
                )
                cv.imwrite(frame_filename, frame)
                frame_count += 1

            current_frame += 1

        video.release()
        cv.destroyAllWindows()
        print(f"Extracted {frame_count} frames to {frames_dir}")
    else:
        print(f"Error: Video file {video_path} does not exist.")


def edge_detection(picture_path):
    img_path = picture_path
    image = cv.imread(img_path)
    (H, W) = image.shape[:2]

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 30, 150)

    start_point1 = (int(W / 2), int(H / 2))
    end_point1 = (W, H)
    cv.line(canny, start_point1, end_point1, 255, 2)

    start_point2 = (int(W / 2), int(H / 2))
    end_point2 = (0, H - 250)
    cv.line(canny, start_point2, end_point2, 255, 2)

    mask1 = np.zeros_like(canny)
    cv.line(mask1, start_point1, end_point1, 255, 2)
    cv.fillPoly(mask1, [np.array([(0, 0), start_point1, end_point1, (W, 0)])], 255)
    canny[mask1 > 0] = 0

    mask2 = np.zeros_like(canny)
    cv.line(mask2, start_point2, end_point2, 255, 2)
    cv.fillPoly(mask2, [np.array([(W, 0), start_point2, end_point2, (0, 0)])], 255)
    canny[mask2 > 0] = 0

    mask_below = np.zeros_like(canny)
    cv.fillPoly(
        mask_below,
        [np.array([[0, H - 250], [int(W / 2), int(H / 2)], [W, H], [0, H]])],
        255,
    )
    # canny[mask_below > 0] = 255

    fill_car(canny, image)

    dpi = 100
    fig_size = (W / dpi, H / dpi)

    plt.figure(figsize=fig_size, dpi=dpi)
    plt.imshow(canny, cmap="gray")
    plt.axis("off")
    plt.savefig("pic/edge2.jpg", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close()

    canny[canny == 255] = 1
    np.savetxt("pic/edge2_values.txt", canny, fmt="%d")

    res = np.multiply(gray, canny)
    plt.imshow(res)
    plt.show()

def fill_car(canny, image):
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create a mask for the filled car region
    car_mask = np.zeros_like(image)

    # Fill the detected contours with a color
    fill_color = (0, 255, 0)  # Green color
    cv.drawContours(car_mask, contours, -1, fill_color, thickness=cv.FILLED)

    # Combine the original image with the mask
    filled_image = cv.addWeighted(image, 1, car_mask, 0.5, 0)

    # Save and display the result
    cv.imwrite("pic/edge_filled.jpg", filled_image)
    cv.imshow("Edge Filled", filled_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Save the edge values
    canny[canny == 255] = 1
    np.savetxt("pic/edge_values.txt", canny, fmt="%d")


def detect_car(picture_path, threshold=0.5):

    frame = cv.imread(picture_path)
    if frame is None:
        raise FileNotFoundError(f"Image not found at path: {picture_path}")

    model_path = r'Checkpoint\car7.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path))

    results = model(frame)
    for result in results.xyxy[0].cpu().numpy():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv.putText(frame, model.names[int(class_id)].upper(), (int(x1), int(y1) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)
            
            # Perform Harris corner detection within the detected car region
            gray = cv.cvtColor(frame[int(y1):int(y2), int(x1):int(x2)], cv.COLOR_BGR2GRAY)
            gray = np.float32(gray)
            corners = cv.cornerHarris(gray, 2, 3, 0.04)
            
            # Dilate corner image to enhance corner points
            corners = cv.dilate(corners, None)
            
            # Threshold for an optimal value, marking the corners
            frame[int(y1):int(y2), int(x1):int(x2)][corners > 0.01 * corners.max()] = [0, 0, 255]
            
            # Optionally, you can draw a small rectangle around each detected corner
            for i in range(corners.shape[0]):
                for j in range(corners.shape[1]):
                    if corners[i, j] > 0.01 * corners.max():
                        cv.rectangle(frame, (int(x1) + j - 5, int(y1) + i - 5), (int(x1) + j + 5, int(y1) + i + 5), (0, 0, 255), 1)

    # Display the result
    cv.imshow('Detected Cars and Corners', frame)
    cv.waitKey(0)
    cv.destroyAllWindows()