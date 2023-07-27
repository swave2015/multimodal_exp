from utils import get_iou, mergeCloseBboxes, merge_overlapping_boxes
from ObjectTracker import ObjectTracker

class TrackerManager:
    def __init__(self):
        self.iou_thres = 0.2
        self.bbox_area_thres = -1
        self.trackers = []

    def update_trackers(self, detection_boxes, keep_counter=1, merge=True):
        if len(detection_boxes) == 0:
            for tracker in self.trackers:
                tracker.objMissingCounter += 1

            new_trackers = []
            for tracker in self.trackers:
                if tracker.objMissingCounter < keep_counter:
                    new_trackers.append(tracker)
            self.trackers = new_trackers  

            return 
        

        if merge:
            detection_boxes = mergeCloseBboxes(detection_boxes)
            detection_boxes = merge_overlapping_boxes(detection_boxes)
        if len(self.trackers) == 0:
            for i, detection_box in enumerate(detection_boxes):
                new_tracker = ObjectTracker(detection_box)
                self.trackers.append(new_tracker)
            return

        # test
        # new_tracker_list = []
        # for i, detection_box in enumerate(detection_boxes):
        #     new_tracker = ObjectTracker(detection_box)
        #     new_tracker_list.append(new_tracker)
        # self.trackers = new_tracker_list
        # return

        detection_to_tracker = {}
        tracker_to_detection = {}
        
        for i, detection_box in enumerate(detection_boxes):
            max_iou = 0
            max_tracker_id = None
            for j, tracker in enumerate(self.trackers):
                iou = get_iou(detection_box, (tracker.x1, tracker.y1, tracker.x2, tracker.y2))
                if iou > max_iou:
                    max_iou = iou
                    max_tracker_id = j
            detection_to_tracker[i] = max_tracker_id
            
        for i, tracker in enumerate(self.trackers):
            max_iou = 0
            max_detection_id = None
            for j, detection_box in enumerate(detection_boxes):
                iou = get_iou(detection_box, (tracker.x1, tracker.y1, tracker.x2, tracker.y2))
                if iou > max_iou and iou > self.iou_thres:
                    max_iou = iou
                    max_detection_id = j
            tracker_to_detection[i] = max_detection_id
        # print('detection_to_tracker: ', detection_to_tracker)
        for detection_id, tracker_id in detection_to_tracker.items():
            if tracker_id is not None and tracker_to_detection.get(tracker_id) == detection_id:
                self.trackers[tracker_id].x1, self.trackers[tracker_id].y1, self.trackers[tracker_id].x2, self.trackers[tracker_id].y2 = detection_boxes[detection_id]
                self.trackers[tracker_id].objMissingCounter = 0
                self.trackers[tracker_id].keepCounter += 1
            else:
                new_tracker = ObjectTracker(detection_boxes[detection_id])
                self.trackers.append(new_tracker)
        # print('tracker_to_detection: ', tracker_to_detection)
        # print('len_self.trackers: ', len(self.trackers))
        for tracker_id, detection_id in tracker_to_detection.items():
            if detection_id is None or detection_to_tracker[detection_id] != tracker_id:
                self.trackers[tracker_id].objMissingCounter += 1
        new_trackers = []
        for tracker in self.trackers:
            if tracker.objMissingCounter < keep_counter:
                new_trackers.append(tracker)
        self.trackers = new_trackers   

    def updateTrackerClipImg(self, img):
        for tracker in self.trackers:
            tracker.update_clip_queue(img, useBG=True)
                