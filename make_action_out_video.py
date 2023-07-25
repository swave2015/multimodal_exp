import cv2
import os

# Directory containing the frames
image_folder = '/data/xcao/code/multimodal_exp/video_out_jpg/9'

# Video file path
video_path = './demo_video_output/output_video.mp4'

target_height = 640
target_width = 640
# Get list of frames sorted by name
images = sorted(os.listdir(image_folder))
if images:
    for image in images:
        # Determine the width and height from the first image
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        height, width, layers = frame.shape
        aspect = width / float(height)
        if aspect > 1:
            # Landscape orientation - wide image
            new_width = target_width
            new_height = int(target_width / aspect)
        else:
            # Portrait orientation - tall image
            new_height = target_height
            new_width = int(target_height * aspect)
        frame_resized = cv2.resize(frame, (new_width, new_height))
        # Padding to get the target dimensions
        top = (target_height - new_height) // 2
        bottom = target_height - new_height - top
        left = (target_width - new_width) // 2
        right = target_width - new_width - left
        frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 15, (target_width, target_height))  # 30 is the fps (frames per second)
        print(image_path)
        video.write(frame_padded)

        # # Write each frame to the video file
        # for image in images:
        #     image_path = os.path.join(image_folder, image)
        #     print(image_path)
        #     video.write(cv2.imread(image_path))

        # Close the video writer
    video.release()
else:
    print("No images found in the directory.")
