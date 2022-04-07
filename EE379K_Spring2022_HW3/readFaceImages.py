import os
import numpy as np
import cv2

def readFaceImages(imdir, ext="png"):
    # imdir: e.g. "hw5_supplemental/faces/"
    # return
    # im    : 2D np.ndarray of images
    # person: np.ndarray of person_id
    # number: np.ndarray of number (lighting condition)
    # subset: np.ndarray of subset_idx
    names = list_files_walk_subdirs(imdir, ext)
    im = []
    person = []
    number = []
    subset = []
    for name in names:
        idx = name.split('/')[-1][6:-4].split('_')
        person.append(int(idx[0]))
        number.append(int(idx[1]))
        image = cv2.imread(name)[:, :, 0]
        im.append(image)
        if number[-1] <= 7:
            subset.append(1)
        elif number[-1] <= 19:
            subset.append(2)
        elif number[-1] <= 31:
            subset.append(3)
        elif number[-1] <= 45:
            subset.append(4)
        elif number[-1] <= 64:
            subset.append(5)
    return np.array(im), np.array(person), np.array(number), np.array(subset)

def list_files_walk_subdirs(path, exts):
    if isinstance(exts, str):
        exts = [exts]
    return [os.path.join(pt, name) for pt, dirs, files in os.walk(path) for name in files if name.lower().split('.')[-1] in exts and (not name.split('/')[-1].startswith('.'))]