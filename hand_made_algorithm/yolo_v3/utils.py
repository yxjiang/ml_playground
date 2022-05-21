import torch

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
