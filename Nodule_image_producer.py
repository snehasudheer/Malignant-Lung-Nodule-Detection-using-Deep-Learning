import glob
import pandas
import ntpath
import numpy
import cv2
import os
from collections import defaultdict


CUBE_IMGTYPE_SRC = "_i"


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]
    cv2.imwrite(target_path, res_img)


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def make_images():
    src_dir = '/Users/macbook/Desktop/Lung Cancer Detection/Input_Coordinates/*'
    dst_dir = '/Users/macbook/Desktop/Lung Cancer Detection/Nodules/'
    new_dir = dst_dir 

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        
        if patient_index == 0:
            new_dir = new_dir + patient_id + '/'
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = load_patient_images(patient_id, "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/", "*" + CUBE_IMGTYPE_SRC + ".png")
        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            #malscore = int(row["malscore"])
            anno_index = row["anno_index"]
            #anno_index = str(anno_index).replace(" ", "xspacex").replace(".", "xpointx").replace("_", "xunderscorex")
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(new_dir + patient_id + "_" + str(anno_index) + "_" + ".png", cube_img, 8, 8)
        print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])



def load_patient_images(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    if base_dir == None:
        base_dir = "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/*"
    src_dir = base_dir + "*" + patient_id + "/"
    print(src_dir)
    src_img_paths = glob.glob(src_dir + wildcard)
    for exclude_wildcard in exclude_wildcards:
        exclude_img_paths = glob.glob(src_dir + exclude_wildcard)
        src_img_paths = [im for im in src_img_paths if im not in exclude_img_paths]
    src_img_paths.sort()
    images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
    images = [im.reshape((1, ) + im.shape) for im in images]
    res = numpy.vstack(images)
    return res


PRINT_TAB_MAP = defaultdict(lambda: [])
def print_tabbed(value_list, justifications=None, map_id=None, show_map_idx=True):
    map_entries = None
    if map_id is not None:
        map_entries = PRINT_TAB_MAP[map_id]

    if map_entries is not None and show_map_idx:
        idx = str(len(map_entries))
        if idx == "0":
            idx = "idx"
        value_list.insert(0, idx)
        if justifications is not None:
            justifications.insert(0, 6)

    value_list = [str(v) for v in value_list]
    if justifications is not None:
        new_list = []
        assert(len(value_list) == len(justifications))
        for idx, value in enumerate(value_list):
            str_value = str(value)
            just = justifications[idx]
            if just > 0:
                new_value = str_value.ljust(just)
            else:
                new_value = str_value.rjust(just)
            new_list.append(new_value)

        value_list = new_list

    line = "\t".join(value_list)
    if map_entries is not None:
        map_entries.append(line)
    print(line)

make_images()



