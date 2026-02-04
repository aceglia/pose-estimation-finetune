import cv2
import os

if __name__ == '__main__':
    video_path = r'D:\Documents\Programmation\pose-estimation-finetune\annotations\data\test\20260204_111129.mp4'
    images_output_dir = os.path.join(os.path.dirname(video_path), 'images')
    os.makedirs(images_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(images_output_dir, f'color_{frame_count}.png'), frame)
        frame_count += 1
    cap.release()

    