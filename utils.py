import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, transform
from sklearn.feature_extraction.image import PatchExtractor

def count_elements_in_folder(folder_path):
    """
    Count the number of elements (files and subfolders) in a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        int: The total number of elements in the folder.
    """
    count = 0
    elements = os.listdir(folder_path)

    for element in elements:
        if os.path.isfile(os.path.join(folder_path, element)):
            count += 1
        else:
            count += count_elements_in_folder(os.path.join(folder_path, element))

    return count


def read_and_convert(img_path):
    """
    Reads an image from the specified path and converts it to grayscale if necessary.

    Args:
        img_path (str): The path of the image file to be read and converted.

    Returns:
        numpy.ndarray: The image represented as a NumPy array, in grayscale.
    """
    img = cv2.imread(img_path)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def process_image(img_path, rescaling_factor=None):
    """
    Process an image by performing histogram equalization and optional rescaling.

    Args:
        img_path (str): The path to the input image file.
        rescaling_factor (float, optional): The factor by which to rescale the image. 
                                            Default is None.

    Returns:
        numpy.ndarray: The processed image.
    """
    img = cv2.imread(img_path)

    if len(img.shape) == 2:
        img = cv2.equalizeHist(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.equalizeHist(img)

    if rescaling_factor is not None:
        img = transform.rescale(img, rescaling_factor)
    
    return img


def extract_patches(img, N, scale=1.0, patch_size=(128, 128), random_state=None):
    """
    Extracts patches from an image.

    Args:
        img (numpy.ndarray): Input image.
        N (int): Maximum number of patches to extract.
        scale (float, optional): Scaling factor for patch size.
                                 Defaults to 1.0.
        patch_size (tuple, optional): Size of the patches to extract.   
                                      Defaults to (128, 128).

    Returns:
        numpy.ndarray: Extracted patches as an array.
    """
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                                max_patches=N,
                                random_state=random_state)
    patches = extractor.transform(img[np.newaxis]).astype(np.uint8)

    for i in range(len(patches)):
        patches[i,:,:] = cv2.equalizeHist(patches[i,:,:])

    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    
    return patches


def sliding_window(img,
                   patch_size=(128, 128),
                   istep=8,
                   jstep=8):
    """
    Generate sliding window patches from an input image.

    Parameters:
        img (numpy.ndarray): Input image.
        patch_size (tuple, optional): Size of the sliding window patch. Default is (128, 128).
        istep (int, optional): Step size in the vertical direction. Default is 8.
        jstep (int, optional): Step size in the horizontal direction. Default is 8.
        scales (numpy.ndarray, optional): Array of scales to apply to the patch size.
            Default is np.linspace(0.5, 2, 4).

    Yields:
        tuple: A tuple containing the patch's top-left corner coordinates (i, j),
            the patch itself after histogram equalization and optional resizing,
            and the applied scale.

    """
    scales = np.linspace(0.5,10,int(10/.5))
    patch_size = np.array(patch_size)
    window_size = np.array(tuple((s * patch_size).astype(int) for s in scales))
    pic_area = img.shape[0]*img.shape[1]
    mask = (window_size[:,0]*window_size[:,1] < 0.25*pic_area) & \
            (window_size[:,0]*window_size[:,1] > 0.01*pic_area)
    filtered_scales = scales[mask]

    while len(filtered_scales)>=6:
        filtered_scales = filtered_scales[::2]
    
    for scale in filtered_scales:
        
        Ni, Nj = (int(scale * s) for s in patch_size)
        
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Nj, jstep):
                
                patch = cv2.equalizeHist(img[i:i + Ni, j:j + Nj])
                
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                
                yield (i, j), patch, scale


def non_max_suppression(boxes, scores, overlapping_threshold=0.5):
    """
    Perform non-maximum suppression on a set of bounding boxes.

    Args:
        boxes (list): List of bounding boxes represented as (x1, y1, x2, y2).
        scores (list): List of corresponding scores for each bounding box.
        overlapping_threshold (float, optional): Threshold for overlapping. Default is 0.7.

    Returns:
        list: List of indices for the selected bounding boxes after non-maximum suppression.

    """

    if len(scores)==0:
        print("There are no detected faces in this image.")
        return

    sorted_indices = np.argsort(scores)[::-1]
    selected_indices = []

    while len(sorted_indices) > 0:

        current_index = sorted_indices[0]
        selected_indices.append(current_index)

        current_box = boxes[current_index]
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        overlaps = []

        for i, index in enumerate(sorted_indices[1:]):

            box = boxes[index]
            area = (box[2] - box[0]) * (box[3] - box[1])

            x1 = max(current_box[0], box[0])
            y1 = max(current_box[1], box[1])
            x2 = min(current_box[2], box[2])
            y2 = min(current_box[3], box[3])

            intersection_box = np.array([x1, y1, x2, y2])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            iou = intersection / (current_area + area - intersection)

            if iou > overlapping_threshold or \
                (intersection_box==box).all() or \
                (intersection_box==current_box).all():
                overlaps.append(i+1)

        sorted_indices = np.delete(sorted_indices, [0] + overlaps)

    return selected_indices


def detect_faces(img_path,
                 model,
                 threshold=0.,
                 patch_size=(128, 128),
                 istep=16,
                 jstep=16,
                 overlapping_threshold=0.6):
    """
    Detect faces in an image using a given model.

    Args:
        img_path (str): Path to the image file.
        model: Model used for face detection.
        threshold (float, optional): Threshold for face classification. Default is 0.
        patch_size (tuple, optional): Size of the patches used for sliding window. Default is (128, 128).
        overlapping_threshold (float, optional): Threshold for overlapping in non-maximum suppression. Default is 0.5.

    Returns:
        numpy.ndarray: Array of selected bounding boxes representing detected faces.

    """
    img = read_and_convert(img_path)
    indices, patches, scales = zip(*sliding_window(img, patch_size=patch_size, istep=istep, jstep=jstep))
    patches_hog = np.array([feature.hog(patch) for patch in patches])
    scores = model.decision_function(patches_hog)
    y_pred = np.where(scores>threshold, 1, 0)
    Ni, Nj = patch_size
    boxes = None
    
    for index in np.where(y_pred==1)[0]:

        x1 = indices[index][1] # x1 = column index
        y1 = indices[index][0] # y1 = row index
        width = int(Nj*scales[index])
        height = int(Ni*scales[index])

        if boxes is None:
            boxes = np.array([[x1, y1, x1+width, y1+height]])
        else:
            new_box = np.array([[x1, y1, x1+width, y1+height]])
            boxes = np.concatenate((boxes, new_box), axis=0)

    selected_indices = non_max_suppression(boxes,
                                           scores[y_pred==1],
                                           overlapping_threshold=overlapping_threshold)
    if selected_indices is None:
        selected_boxes = []
        selected_scores = []
    else:
        selected_boxes = boxes[selected_indices]
        selected_scores = scores[y_pred==1][selected_indices]
    
    return selected_boxes, selected_scores