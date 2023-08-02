from datetime import timedelta

# ... (other parts of your script here)

# Assume 25 frames per second
frame_rate = 25

# Initialize an empty list to hold all the subtitles
subtitles = []

# Write each frame to the video file
for frame_number, image in enumerate(images):
    image_path = os.path.join(image_folder, image)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    label_file_name = image.replace('.jpg', '.txt')
    label_file_path = os.path.join(image_folder, label_file_name)
    if os.path.exists(label_file_path):
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # ... (other parts of your script here)
                if caption != 'recognizing':
                    img = caption_multi_line((x1, y1), caption, img, 
                                             caption_font, rgb_color, (0, 0), 
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
    # ... (other parts of your script here)

# ... (other parts of your script here)

# After generating the video, write the subtitles to the .srt file
with open("subtitles.srt", "w") as f:
    for subtitle in subtitles:
        f.write(str(subtitle["index"]) + "\n")
        f.write(subtitle["start"] + " --> " + subtitle["end"] + "\n")
        f.write(subtitle["caption"] + "\n\n")
