import cv2
import os

video_path = '/data/xcao/code/multimodal_exp/demo_video_output'
for dir_name in os.listdir(video_path):
    video_save_path = os.path.join(video_path, dir_name + '.mp4')
    image_folder = os.path.join(video_path, dir_name, 'demo_video')
    # Video file path
    # Get list of frames sorted by name
    images = sorted(os.listdir(image_folder))
    if images:
        # Determine the width and height from the first image
        image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(image_path)
        height, width, layers = frame.shape

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_save_path, fourcc, 15, (width, height))  # 30 is the fps (frames per second)

        # Write each frame to the video file
        for image in images:
            image_path = os.path.join(image_folder, image)
            print(image_path)
            video.write(cv2.imread(image_path))

        # Close the video writer
        video.release()
    else:
        print("No images found in the directory.")
