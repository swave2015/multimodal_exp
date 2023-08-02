import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from datetime import timedelta



def caption_multi_line(xy, caption, img, caption_font, rgb_color, xy_shift, isBbox=False, split_len=6):
    text_color = (0, 0, 0)
    x1, y1 = xy
    x1_shift, y1_shift = xy_shift
    split_lines = caption.split(' ')
    lines = []
    draw = ImageDraw.Draw(img)
    if int(len(split_lines) / split_len) == 0:
        if isBbox:
            text_size = caption_font.getsize(caption)
            draw.rectangle([x1 + x1_shift, y1 + y1_shift - text_size[1], x1 + x1_shift + text_size[0], y1 + y1_shift], fill=rgb_color)
            draw.text((x1 + x1_shift, y1 + y1_shift - caption_font.getsize(caption)[1]), caption, font=caption_font, fill=text_color)
        else:
            text_size = caption_font.getsize(caption)
            draw.rectangle([x1, y1, x1 + text_size[0], y1 + text_size[1]], fill=rgb_color)
            draw.text((x1, y1), caption, font=caption_font, fill=text_color)
        return img
    elif int(len(split_lines) / split_len) > 0:
        for i in range(math.ceil(len(split_lines) / split_len)):
            lines.append(split_lines[split_len * i: split_len * (i + 1)])
    y_text = y1
    x_text = x1
    line_show_list = []
    max_x_size = 0
    y_text_height = 0
    for line in lines:
        line_show = ' '.join(line)
        line_show_list.append(line_show)
        text_size = caption_font.getsize(line_show)
        x_text_size = text_size[0]
        y_text_height += text_size[1]
        if x_text_size > max_x_size:
            max_x_size = x_text_size

    if isBbox:
        y_text = y1 - y_text_height + y1_shift
        x_text = x1  + x1_shift
    
  

    draw.rectangle([x_text, y_text, x_text + max_x_size, y_text + y_text_height], fill=rgb_color)

    for line in line_show_list:
        draw.text((x_text, y_text), line, font=caption_font, fill=text_color)
        y_text += caption_font.getsize(line)[1]
    
    return img

def draw_thick_rectangle(draw, coords, color, thickness):
    for i in range(thickness):
        rect_start = (coords[0] - i, coords[1] - i)
        rect_end = (coords[2] + i, coords[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color)



# Directory containing the frames
image_folder = '/data/xcao/code/multimodal_exp/video_out_jpg/black_bear/demo_video'
caption_font = ImageFont.truetype("/data/xcao/code/multimodal_exp/miscellaneous/fonts/Arial.ttf", 30)
# Video file path
video_path = '/data/xcao/code/multimodal_exp/demo_video_output/output_video.mp4'
rgb_color = (84, 198, 247)
frame_rate = 15
subtitles = []
# Get list of frames sorted by name
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
if images:
    # Determine the width and height from the first image
    image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))  # 30 is the fps (frames per second)

    # Write each frame to the video file
    for frame_number, image in enumerate(images):
        if frame_number >= 1645:
            break
        image_path = os.path.join(image_folder, image)
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        label_file_name = image.replace('.jpg', '.txt')
        label_file_path = os.path.join(image_folder, label_file_name)
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    x1, y1, x2, y2 = map(int, line.strip().split(';')[:-1])
                    # if x1 > 850 and x2 < 990:
                    #     continue
                    # if x1 > 610 and x2 < 680:
                    #     continue
                    # if x1 > 870 and x2 < 950:
                    #     continue
                    # if x1 > 950 and x2 < 1100:
                    #     continue
                    # if x1 > 700 and x2 < 800:
                    #     continue
                    area = (x2 - x1) * (y2 - y1)
                    if area < (899 - 725) * (371 - 213) / 2:
                        continue
                    caption = line.strip().split(';')[-1]
                    draw_thick_rectangle(draw, [x1, y1, x2, y2], rgb_color, 5)
                    # img = caption_multi_line((x1, y1), caption + '_' + str(image), img, 
                    #                                 caption_font, rgb_color, (0, 0), 
                    #                                 isBbox=True, split_len=20)
                    if caption != 'recognizing' and frame_number <= 670:
                        img = caption_multi_line((x1, y1), caption, img, 
                                                    caption_font, rgb_color, (-4, 0), 
                                                    isBbox=True, split_len=20)

                        # Add the subtitle to the list
                        start_time = timedelta(seconds=frame_number/frame_rate)
                        end_time = timedelta(seconds=(frame_number+1)/frame_rate)  # Assume each caption lasts for 1 frame
                        subtitles.append({
                            "index": len(subtitles) + 1,
                            "start": str(start_time),
                            "end": str(end_time),
                            "caption": caption
                        })

                        
        print(image_path)
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        video.write(opencv_image)

    # Close the video writer
    video.release()

    with open("subtitles.srt", "w") as f:
        for subtitle in subtitles:
            f.write(str(subtitle["index"]) + "\n")
            f.write(subtitle["start"] + " --> " + subtitle["end"] + "\n")
            f.write(subtitle["caption"] + "\n\n")
else:
    print("No images found in the directory.")
