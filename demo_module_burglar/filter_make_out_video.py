import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from datetime import timedelta



def caption_multi_line(xy, translations, caption, img, caption_font, caption_font_Chinese, rgb_color, xy_shift, isBbox=False, split_len=6):
    text_color = (0, 0, 0)
    x1, y1 = xy
    x1_shift, y1_shift = xy_shift
    split_lines = caption.split(' ')
    lines = []
    draw = ImageDraw.Draw(img)
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw_transp = ImageDraw.Draw(transp, "RGBA")
    if int(len(split_lines) / split_len) == 0:
        if isBbox:
            text_size = caption_font.getsize(caption)
            text_size_trans = caption_font_Chinese.getsize(translations[caption])
            draw_transp.rectangle([x1 + x1_shift, y1 + y1_shift - text_size[1] - text_size_trans[1] -5 , x1 + x1_shift + max(text_size[0], text_size_trans[0]), y1 + y1_shift], fill=rgb_color)
            img.paste(Image.alpha_composite(img, transp))
            draw = ImageDraw.Draw(img)
            draw.text((x1 + x1_shift, y1 + y1_shift - caption_font_Chinese.getsize(caption)[1] - caption_font.getsize(caption)[1]), translations[caption], font=caption_font_Chinese, fill=text_color)
            draw.text((x1 + x1_shift, y1 + y1_shift - 5 - caption_font.getsize(caption)[1]), caption, font=caption_font, fill=text_color)
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
image_folder = '/data/xcao/code/multimodal_exp/video_out_jpg/burglars/demo_video'
caption_font = ImageFont.truetype("/data/xcao/code/multimodal_exp/miscellaneous/fonts/Arial.ttf", 50)
caption_font_Chinese = ImageFont.truetype("/data/xcao/code/multimodal_exp/miscellaneous/fonts/weiruanyahei.ttf", 45)
# Video file path
video_path = '/data/xcao/code/multimodal_exp/demo_video_output/output_video.mp4'
rgb_color = (84, 198, 247)
rgb_color_with_alpha = (84, 198, 247, 150)
frame_rate = 30
subtitles = []
# Get list of frames sorted by name
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
caption_set = set()

translations = {
    'cleaning the window': '正在清洁窗户',
    'burglar': '小偷',
    'they are trying to open the window': '他们正试图打开窗户',
    'trying to get into the house': '正在试图进入房子',
    'breaking into a house': '正在闯入房屋',
    'he is cleaning the window': '他正在清洁窗户',
    'he is trying to get into the house': '他正在试图进入房子',
    'the person is using a hose to clean the outside of the house': '这个人正在使用水管清洗房子外墙',
    'they are trying to break into the house': '他们正在试图闯入房子',
    'burgling': '正在偷盗',
    'they are trying to open the door': '他们正在试图打开门',
    'he is trying to open the door': '他正在试图打开门',
    'vacuuming': '正在使用吸尘器清扫',
    'trying to break into the house': '正在试图闯入房子'
    }

if images:
    # Determine the width and height from the first image
    image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))  # 30 is the fps (frames per second)

    # Write each frame to the video file
    last_caption = ''
    start_time = ''
    end_time = ''
    frame_caption_list = []
    for frame_number, image in enumerate(images):

        start_time = timedelta(seconds=frame_number/frame_rate)
        seconds, microseconds = divmod(start_time.microseconds, 1000) 
        start_time = f"{str(start_time).split('.')[0]}.{microseconds:03d}"

        end_time = timedelta(seconds=(frame_number+1)/frame_rate)  # Assume each caption lasts for 1 frame
        seconds_end, microseconds_end = divmod(end_time.microseconds, 1000) 
        end_time = f"{str(end_time).split('.')[0]}.{microseconds_end:03d}"

        image_path = os.path.join(image_folder, image)
        img = Image.open(image_path).convert('RGBA')
        draw = ImageDraw.Draw(img)
        label_file_name = image.replace('.jpg', '.txt')
        label_file_path = os.path.join(image_folder, label_file_name)
        total_caption = ''
        
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
                    if area < (554 - 432) * (554 - 260) / 10:
                        continue
                    caption = line.strip().split(';')[-1]
                    draw_thick_rectangle(draw, [x1, y1, x2, y2], rgb_color, 5)
                    # img = caption_multi_line((x1, y1), caption + '_' + str(image), img, 
                    #                                 caption_font, rgb_color, (0, 0), 
                    #                                 isBbox=True, split_len=20)
                    if caption != 'recognizing':
                        img = caption_multi_line((x1, y1), translations, caption, img, 
                                                    caption_font, caption_font_Chinese, rgb_color_with_alpha, (-4, 0), 
                                                    isBbox=True, split_len=20)

                        # Add the subtitle to the list
                        
                        if total_caption != '':
                            # total_caption += '\n' + translations[caption]
                            total_caption += '\n' + caption
                        else:
                            # total_caption = translations[caption]
                            total_caption = caption

        frame_caption_list.append(total_caption)



        if total_caption != '':
            if len(subtitles) > 0:
                if total_caption == last_caption and frame_number > 0 and frame_caption_list[frame_number -1] != '':
                    subtitles[-1]["end"] = str(end_time)
                    # subtitles.append({
                    #     "index": len(subtitles) + 1,
                    #     "start": str(start_time),
                    #     "end": str(end_time),
                    #     "caption": caption
                    # })
                else:
                    subtitles.append({
                        "index": len(subtitles) + 1,
                        "start": str(start_time),
                        "end": str(end_time),
                        "caption": total_caption
                    })
                    last_caption = total_caption
            else:
                subtitles.append({
                    "index": len(subtitles) + 1,
                    "start": str(start_time),
                    "end": str(end_time),
                    "caption": total_caption
                })
                last_caption = total_caption


        caption_set.add(total_caption)

                        
        print(image_path)
        img.convert('RGB')
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        # video.write(opencv_image)

    # Close the video writer
    video.release()

    with open("subtitles.srt", "w", encoding="utf-8") as f:
        for index, subtitle in enumerate(subtitles):
            f.write(str(subtitle["index"]) + "\n")
            f.write(subtitle["start"] + " --> " + subtitle["end"] + "\n")
            f.write(subtitle["caption"] + "\n\n")

    # caption_list = list(caption_set)
    # caption_file_path = "caption_list.txt"
    # with open(caption_file_path, "w") as file:
    #     for item in caption_list:
    #         file.write(f"{item}\n")


else:
    print("No images found in the directory.")
