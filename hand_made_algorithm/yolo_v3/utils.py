import torch
from collections import Counter

def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union.

    Parameters:
        boxes_preds (tensor): Predictions of bounding boxes (N x 4).
        boxes_labels (tensor): Ground truth of bounding boxes (N x 4).
    
    Returns:
        tensor: IoU of all instances.
    """
    # Use slicing to keep the dimensions of the tensor
    if box_format == "midpoint":  # mid_x, mid_y, w, h
        preds_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        preds_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        preds_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        preds_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        labels_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        labels_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        labels_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        labels_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
    elif box_format == "corners": # x1, y1, x2, y2
        preds_x1 = boxes_preds[..., 0:1]
        preds_y1 = boxes_preds[..., 1:2]
        preds_x2 = boxes_preds[..., 2:3]
        preds_y2 = boxes_preds[..., 3:4]
        labels_x1 = boxes_labels[..., 0:1]
        labels_y1 = boxes_labels[..., 1:2]
        labels_x2 = boxes_labels[..., 2:3]
        labels_y2 = boxes_labels[..., 3:4]

    top_left_x = torch.max(preds_x1, labels_x1)
    top_left_y = torch.max(preds_y1, labels_y1)
    bottom_right_x = torch.min(preds_x2, labels_x2)
    bottom_right_y = torch.min(preds_y2, labels_y2)
    # Also handle the edge case that there is no intersection.
    intersection = (bottom_right_x - top_left_x).clamp(0) * (bottom_right_y - top_left_y).clamp(0)

    area1 = abs((preds_x2 - preds_x1) * (preds_y2 - preds_y1))
    area2 = abs((labels_x2 - labels_x1) * (labels_y2 - labels_y1))
    return intersection / (area1 + area2 - intersection)


def nms(predictions, iou_threshold, prob_threshold, box_format="corners"):
    """
    1. Discard bounding boxes < probability threshold.
    2. For each object, pick the bounding boxes with the highest probability and remove meaningfully overlapped ones.
    """
    # predictions = [[class_idx, prob_bbox, x1, y1, x2, y2], [...], ...]
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes_after_nms = []
    # sort bboxes with the highest prob at the beginning
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)

    while bboxes:
        chosen_box = bboxes.pop(0)
        # keep the boxes either not with the same class or with less overlapping
        bboxes = [
            box 
            for box in bboxes
            if box[0] != chosen_box[0] or 
                intersection_over_union(
                    torch.tensor(chosen_box[2:]), 
                    torch.tensor(box[2:]), 
                    box_format=box_format
                ) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms


def mean_average_precision(detected_boxes, true_boxes, iou_thresholds, box_format="corners", num_classes=20):
    """
    Calculate the mean average precision. The process is as follows:
    1. For each class, rank the confidence from high to low, and then calculate accumulative precision and recall.
    2. Calculate the AUC for each class.
    3. Get the average precision by calculating the average AUC for all classes.
    4. Get the mAP by calculate the mean of all such average precision with different IoU threshold.

    detected_boxes and true_boxes: list of 7-tuples: [train_idx, class_pred, prob, x1, y1, x2, y2]
    """
    return sum([
            average_precisions(detected_boxes, true_boxes, iou_threshold, box_format, num_classes) 
            for iou_threshold in iou_thresholds
           ]) / len(iou_thresholds)


def average_precisions(detected_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20)
    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detected = []
        ground_truths = []

        # Collect the boxes of the specified class. 
        for bbox in detected_boxes:
            if bbox[0] == c:
                detected.append(bbox)
        for bbox in true_boxes:
            if bbox[0] == c:
                detected.append(bbox)
        # Group by image.
        amount_bboxes = Counter([bbox[0] for bbox in ground_truths])

        # Create tensor with the dimension equal to the number of bboxes of class c in each image.
        for k, v in amount_bboxes.items():
            amount_bboxes[k] = torch.zeros(v)

        # Calculate the AUC.
        detected.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detected)))
        FP = torch.zeros((len(detected)))
        num_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detected):
            ground_truth_imgs = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            num_ground_truths = len(ground_truth_imgs)
            best_iou = 0
            best_ground_truth_idx = 0
            # Find the best matched ground trueth of the target detected bbox.
            for idx, gt in enumerate(ground_truth_imgs):
                iou = intersection_over_union(torch.tensor(detection[3:1]), torch.tensor(gt[3:]), box_format=box_format)
                if iou > best_iou:
                    best_iou = best_iou
                    best_ground_truth_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_ground_truth_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_ground_truth_idx] = 1
                else:  # The ground truth bbox is already covered.
                    FP[detection_idx] = 1
            else:  # No match.
                FP[detection_idx] = 1
        
        # [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (num_true_bboxes + epsilon)
        recalls = torch.cat((torch.tensors[0]), recalls)  # Append an 0 at the end.
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))  # Append an 1 at the end.
        average_precisions. append(torch.trapz(precisions, recalls))
    
    return sum(average_precisions) / len(average_precisions)