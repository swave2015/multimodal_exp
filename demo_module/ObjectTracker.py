import random
from collections import deque
import cv2
from collections import Counter

class ObjectTracker:
    def __init__(self, bbox, capQueuLen=10):
        self.x1 = bbox[0]
        self.y1 = bbox[1]
        self.x2 = bbox[2]
        self.y2 = bbox[3]
        self.id = random.randint(0, 100)
        self.objMissingCounter = 0
        self.keepCounter = 1
        self.clipQueue = deque(maxlen=9)
        self.captionQueue = deque(maxlen=capQueuLen)
        self.clipImg = None
        self.enlarge_factor = 1.2
        self.caption_show = 'recognizing'
        self.caption_last_infer = None
        self.caption_keep_counter = 0

    def find_max_frequency_caption(self):
        if len(self.captionQueue) < self.captionQueue.maxlen:
            return
        # Count the occurrences of each caption in captionQueue
        caption_counts = Counter(self.captionQueue)
        
        # Find the caption with the maximum frequency
        # most_common() returns a list of tuples (element, count), where the first element has the highest count
        most_common_caption = caption_counts.most_common(1)
        
        # Check if the list is not empty
        if most_common_caption:
            # Return the caption with the highest frequency
            self.caption_show = most_common_caption[0][0]
        else:
            return

    def update_clip_queue(self, frame):
        box_width = self.x2 - self.x1
        box_height = self.y2 - self.y1
        frame_h, frame_w, _ = frame.shape
        side_len = min(max(box_width, box_height) \
                       , min(frame_h, frame_w))
        side_len *= self.enlarge_factor
        center_x = (self.x1 + self.x2) / 2
        center_y = (self.y1 + self.y2) / 2
        start_x = max(0, center_x - side_len // 2)
        start_y = max(0, center_y - side_len // 2)
        end_x = min(start_x + side_len, frame_w - 1)
        end_y = min(start_y + side_len, frame_h - 1)
        crop_img = frame[int(start_y):int(end_y), int(start_x):int(end_x)]
        if crop_img.size > 0:
            self.clipQueue.append(crop_img)
        if len(self.clipQueue) >= 9:
            max_width, max_height = 0, 0
            for img in self.clipQueue:
                height, width = img.shape[:2]
                max_width, max_height = max(max_width, width), max(max_height, height)
            resized_imgs = []
            for index, img in enumerate(self.clipQueue):
                height, width = img.shape[:2]
                scale_ratio_w = max_width / width
                scale_ratio_h = max_height / height
                if width * scale_ratio_h <= max_width:
                    scale_ratio = scale_ratio_h
                else:
                    scale_ratio = scale_ratio_w  
                new_dims = (round(width*scale_ratio), round(height*scale_ratio))
                resized_img = cv2.resize(img, new_dims, interpolation=cv2.INTER_AREA)
                pad_color = 0 
                padded_img = cv2.copyMakeBorder(resized_img, 
                                            top=0, 
                                            bottom=max_height - new_dims[1], 
                                            left=0, 
                                            right=max_width - new_dims[0], 
                                            borderType=cv2.BORDER_CONSTANT, 
                                            value=pad_color)
                p_h, p_w = padded_img.shape[:2]
                resized_imgs.append(padded_img)
            horizontal_imgs = []
            for i in range(0, len(resized_imgs), 3):
                concat_img = cv2.hconcat([resized_imgs[i], resized_imgs[i+1], resized_imgs[i+2]])
                horizontal_imgs.append(concat_img)
            self.clipImg = cv2.vconcat(horizontal_imgs)
            self.clipQueue.popleft()
        

    def get_position(self):
        """Get current position of the object."""
        return (self.x1, self.y1, self.x2, self.y2)
