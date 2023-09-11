import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import math
from datetime import timedelta
import jieba

def caption_multi_line(xy, translations, caption, img, caption_font, caption_font_Chinese, rgb_color, xy_shift, isBbox=False, split_len=20, split_len_chinese=20):
    text_color = (0, 0, 0)
    x1, y1 = xy
    x1_shift, y1_shift = xy_shift
    split_lines = caption.split(' ')
    split_lines_translation = caption.split(' ')
    # split_lines_translation = jieba.lcut(translations[caption])
    lines = []
    draw = ImageDraw.Draw(img)
    transp = Image.new('RGBA', img.size, (0,0,0,0))
    draw_transp = ImageDraw.Draw(transp, "RGBA")
    Chinese_index = 0
    if int(len(split_lines) / split_len) == 0:
        if isBbox:
            text_size = caption_font.getsize(caption)
            text_size_trans = caption_font_Chinese.getsize(translations.get(caption, caption))
            # text_size_trans = caption_font_Chinese.getsize(caption)
            draw_transp.rectangle([x1 + x1_shift, y2, x1 + x1_shift + max(text_size[0], text_size_trans[0]), y2 + y1_shift + y1_shift + text_size[1] + text_size_trans[1] + 5], fill=rgb_color)
            img.paste(Image.alpha_composite(img, transp))
            draw = ImageDraw.Draw(img)
            # draw.text((x1 + x1_shift, y1 + y1_shift - caption_font_Chinese.getsize(caption)[1] - caption_font.getsize(caption)[1]), translations[caption], font=caption_font_Chinese, fill=text_color)
            # draw.text((x1 + x1_shift, y1 + y1_shift - 15 - caption_font_Chinese.getsize(translations.get(caption, caption))[1] - caption_font.getsize(translations.get(caption, caption))[1]), translations.get(caption, caption), font=caption_font_Chinese, fill=text_color)
            # draw.text((x1 + x1_shift, y1 + y1_shift - 5 - caption_font.getsize(caption)[1]), caption, font=caption_font, fill=text_color)
            draw.text((x1 + x1_shift, y2), caption, font=caption_font, fill=text_color)
            draw.text((x1 + x1_shift, y2 + y1_shift + caption_font.getsize(caption)[1]), translations.get(caption, caption), font=caption_font_Chinese, fill=text_color)
        else:   
            text_size = caption_font.getsize(caption)
            draw.rectangle([x1, y1, x1 + text_size[0], y1 + text_size[1]], fill=rgb_color)
            draw.text((x1, y1), caption, font=caption_font, fill=text_color)
        return img
    elif int(len(split_lines) / split_len) > 0:
        for i in range(math.ceil(len(split_lines_translation) / split_len_chinese)):
            lines.append(split_lines_translation[split_len_chinese * i: split_len_chinese * (i + 1)])
        Chinese_index = len(lines)
        for i in range(math.ceil(len(split_lines) / split_len)):
            lines.append(split_lines[split_len * i: split_len * (i + 1)])
    y_text = y1
    x_text = x1
    line_show_list = []
    max_x_size = 0
    y_text_height = 0
    for index, line in enumerate(lines):
        if index < Chinese_index:
            line_show = ''.join(line)
            line_show_list.append(line_show)
            text_size = caption_font_Chinese.getsize(line_show)
            x_text_size = text_size[0]
            y_text_height += text_size[1]
            if x_text_size > max_x_size:
                max_x_size = x_text_size
        else:
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
    
    # draw.rectangle([x_text, y_text, x_text + max_x_size, y_text + y_text_height], fill=rgb_color)
    text_size_trans = caption_font_Chinese.getsize(translations.get(caption, caption))
    # draw_transp.rectangle([x1 + x1_shift, y1 + y1_shift - text_size[1] - text_size_trans[1] -5 , 
    #                        x1 + x1_shift + max_x_size, y1 + y1_shift], 
    #                        fill=rgb_color)
    draw_transp.rectangle([x1 + x1_shift, y_text, 
                           x1 + x1_shift + max_x_size, y1 + y1_shift], 
                           fill=rgb_color)
    
    img.paste(Image.alpha_composite(img, transp))
    draw = ImageDraw.Draw(img)

    for index, line in enumerate(line_show_list):
        if index == 0:
            y_text = y_text - 5
        if index < Chinese_index:
            draw.text((x_text, y_text), line, font=caption_font_Chinese, fill=text_color)
        else:
            draw.text((x_text, y_text), line, font=caption_font, fill=text_color)
        if index == Chinese_index - 1:
            y_text += caption_font.getsize(line)[1] + 10
        elif index < Chinese_index - 1:
            y_text += caption_font_Chinese.getsize(line)[1]
        else:
            y_text += caption_font.getsize(line)[1]

    # for line in line_show_list:
    #     draw.text((x_text, y_text), line, font=caption_font, fill=text_color)
    #     y_text += caption_font.getsize(line)[1]
    
    return img

def draw_thick_rectangle(draw, coords, color, thickness):
    for i in range(thickness):
        rect_start = (coords[0] - i, coords[1] - i)
        rect_end = (coords[2] + i, coords[3] + i)
        draw.rectangle((rect_start, rect_end), outline=color)



def format_timedelta(t):
    """Return a string representation of a timedelta rounded to three decimal places."""
    hours, remainder = divmod(t.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:06.3f}'.format(int(hours), int(minutes), seconds)

# Directory containing the frames
image_folder = '/data/xcao/code/multimodal_exp/video_out_jpg/black_bear/demo_video'
caption_font = ImageFont.truetype("/data/xcao/code/multimodal_exp/miscellaneous/fonts/Arial.ttf", 45)
caption_font_Chinese = ImageFont.truetype("/data/xcao/code/multimodal_exp/miscellaneous/fonts/weiruanyahei.ttf", 45)
# Video file path
video_path = '/data/xcao/code/multimodal_exp/demo_video_output/output_video.mp4'
rgb_color = (84, 198, 247)
rgb_color_with_alpha = (84, 198, 247, 150)
frame_rate = 15
subtitles = []
# Get list of frames sorted by name
images = sorted([img for img in os.listdir(image_folder) if img.endswith(".jpg")])
caption_set = set()

translations = {
    'he\'s trying to get into the birdhouse': '他正在试图进入鸟屋',
    'he is trying to get into the chicken coop': '他正在试图进入鸡舍',
    'he\'s trying to get into the chicken coop': '他正在试图进入鸡舍',
    'eating bird food': '吃鸟食',
    'trying to get into the bird feeder': '试图进入喂鸟器',
    'he is trying to get into the bird house': '他正在试图进入鸟屋',
    'trying to get into the chicken coop': '试图进入鸡舍',
    'he is trying to get into the bird feeder': '他正在试图进入喂鸟器',
    'looking for food': '寻找食物'
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

    # os.remove('./time_debug.txt')

    for frame_number, image in enumerate(images):
        delta_number = 0
        if frame_number >= 1645:
            break
        frame_number = frame_number - delta_number
        start_time = timedelta(seconds=frame_number/frame_rate)
        start_time_ori = start_time
        start_time = format_timedelta(start_time)

        end_time = timedelta(seconds=(frame_number+1)/frame_rate)  # Assume each caption lasts for 1 frame
        end_time_ori = end_time
        end_time = format_timedelta(end_time)

        # with open("time_debug.txt", "a", encoding="utf-8") as f:
        #     f.write('frame_number: ' + str(frame_number)
        #             + '_start_time: ' 
        #             + start_time
        #             + '_end_time: '
        #             + end_time
        #             + '_start_time_ori: ' 
        #             + str(start_time_ori)
        #             + '_end_time_ori: '
        #             + str(end_time_ori)
        #             + "\n")
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
                    if area < (899 - 725) * (371 - 213) / 2:
                        continue
                    caption = line.strip().split(';')[-1]
                    draw_thick_rectangle(draw, [x1, y1, x2, y2], rgb_color, 5)
                    # img = caption_multi_line((x1, y1), caption + '_' + str(image), img, 
                    #                                 caption_font, rgb_color, (0, 0), 
                    #                                 isBbox=True, split_len=20)
                    if caption != 'recognizing' and frame_number <= 670:
                        img = caption_multi_line((x1, y1), translations, caption, img, 
                                                    caption_font, caption_font_Chinese, rgb_color_with_alpha, (-4, 0), 
                                                    isBbox=True, split_len=20, split_len_chinese=20)

                        # Add the subtitle to the list
                        
                        if total_caption != '':
                            total_caption += '\n' + translations.get(caption, caption)
                            # total_caption += '\n' + caption
                        else:
                            total_caption = translations.get(caption, caption)
                            # total_caption = caption

        frame_caption_list.append(total_caption)
        caption_set.add(total_caption)


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

                        
        print(image_path)
        img.convert('RGB')
        numpy_image = np.array(img)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
        video.write(opencv_image)

    # Close the video writer
    video.release()

    with open("subtitles.srt", "w", encoding="utf-8") as f:
        for index, subtitle in enumerate(subtitles):
            f.write(str(subtitle["index"]) + "\n")
            f.write(subtitle["start"] + " --> " + subtitle["end"] + "\n")
            f.write(subtitle["caption"] + "\n\n")

    caption_list = list(caption_set)
    caption_file_path = "caption_list.txt"
    with open(caption_file_path, "w") as file:
        for item in caption_list:
            file.write(f"{item}\n")

else:
    print("No images found in the directory.")
