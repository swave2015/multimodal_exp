from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial import distance
from PIL import Image, ImageDraw, ImageFont
import math

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float in [0, 1]
        The Intersection over Union (IoU) between the two bounding boxes
    """
    intersect_area = max(0, min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])) * max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersect_area / float(bb1_area + bb2_area - intersect_area)
    return iou

def mergeCloseBboxes(target_boxes, distance_factor=10):
    def distance_measure(box1, box2):
        # Generate coordinates for the four corners of each box
        corners1 = [(box1[0], box1[1]), (box1[0], box1[3]), (box1[2], box1[1]), (box1[2], box1[3])]
        corners2 = [(box2[0], box2[1]), (box2[0], box2[3]), (box2[2], box2[1]), (box2[2], box2[3])]
        
        # Compute the Euclidean distance between each pair of corners and return the minimum
        min_distance = min(distance.euclidean(corner1, corner2) for corner1 in corners1 for corner2 in corners2)
        return min_distance
    if len(target_boxes) <= 0:
        return []
    # Generate feature vectors for each bounding box, consisting of its four sides
    features = np.array([[box[0], box[1], box[2], box[3]] for box in target_boxes])

    # Perform DBSCAN clustering based on the distances between the feature vectors
    clustering = DBSCAN(eps=distance_factor, min_samples=1, metric=distance_measure).fit(features)
    
    labels = clustering.labels_
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    merged_boxes = []
    for cluster_idx in range(num_clusters):
        indices = np.where(clustering.labels_ == cluster_idx)[0]
        cluster_boxes = [target_boxes[i] for i in indices]
        
        # merge boxes in the same cluster to a single box by finding the min and max coordinates
        x1 = min(box[0] for box in cluster_boxes)
        y1 = min(box[1] for box in cluster_boxes)
        x2 = max(box[2] for box in cluster_boxes)
        y2 = max(box[3] for box in cluster_boxes)
        merged_boxes.append([x1, y1, x2, y2])

    return merged_boxes

def merge_overlapping_boxes(boxes):
    # Put the intersecting boxes into the same group
    grouped_boxes = []
    while len(boxes) > 0:
        box_group = [boxes[0]]
        del boxes[0]
        compare_index = 0
        while compare_index < len(box_group):
            i = 0
            while i < len(boxes):
                if get_iou(box_group[compare_index], boxes[i]) > 0:
                    box_group.append(boxes[i])
                    del boxes[i]
                else:
                    i += 1
            compare_index += 1
        grouped_boxes.append(box_group)

    # Merge boxes in the same group into a single large box
    merged_boxes = []
    for group in grouped_boxes:
        if len(group) <= 0:
            continue
        box0 = group[0]
        xmin, ymin, xmax, ymax = box0[0], box0[1], box0[2], box0[3]
        for box in group:
            xmin = min(xmin, box[0])
            xmax = max(xmax, box[2])
            ymin = min(ymin, box[1])
            ymax = max(ymax, box[3])
        merged_box = [xmin, ymin, xmax, ymax]
        merged_boxes.append(merged_box)

    return merged_boxes

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



# def merge_overlapping_boxes(boxes):
#     merged_boxes = []
#     while boxes:
#         # Start with the first box
#         base_box = boxes.pop(0)
#         base_box = list(base_box)  # Make sure base_box is mutable

#         to_delete = []  # This will hold the indices of boxes to be deleted
#         for i, box in enumerate(boxes):
#             if get_iou(base_box, box) > 0:
#                 # If boxes overlap, merge them
#                 base_box[0] = min(base_box[0], box[0])
#                 base_box[1] = min(base_box[1], box[1])
#                 base_box[2] = max(base_box[2], box[2])
#                 base_box[3] = max(base_box[3], box[3])
#                 to_delete.append(i)

#         # Delete merged boxes from the boxes list
#         for index in sorted(to_delete, reverse=True):
#             del boxes[index]

#         merged_boxes.append(base_box)

#     return merged_boxes


def resize_or_pad(input_path, output_path, target_size):
    """
    Resize the image if it's larger than the target size. 
    If it's smaller, pad it to the target size.
    
    Parameters:
    - input_path: path to the original image.
    - output_path: path to save the resized or padded image.
    - target_size: desired size for the output image.
    """
    img = cv2.imread(input_path)
    
    if img.shape[0] > target_size or img.shape[1] > target_size:
        img_resized = cv2.resize(img, (target_size, target_size))
        cv2.imwrite(output_path, img_resized)
    else:
        top = (target_size - img.shape[0]) // 2
        bottom = target_size - img.shape[0] - top
        left = (target_size - img.shape[1]) // 2
        right = target_size - img.shape[1] - left
        img_padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        cv2.imwrite(output_path, img_padded)

# Example usage
resize_or_pad("path_to_original_image.jpg", "path_to_output_image.jpg", 300)
