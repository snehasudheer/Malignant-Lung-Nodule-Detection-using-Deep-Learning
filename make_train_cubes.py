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


def make_annotation_images_lidc():
    src_dir = '/Users/macbook/Desktop/Lung Cancer Detection/Extracted_Images/*'

    dst_dir = '/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_lidc/'

    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)
    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos_lidc.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annos_pos_lidc.csv", "")
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = load_patient_images(patient_id, "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/", "*" + CUBE_IMGTYPE_SRC + ".png")
        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            malscore = int(row["malscore"])
            anno_index = row["anno_index"]
            anno_index = str(anno_index).replace(" ", "xspacex").replace(".", "xpointx").replace("_", "xunderscorex")
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore * malscore) + "_1_pos.png", cube_img, 8, 8)
        print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_pos_annotation_images_manual():
    src_dir = "/Users/macbook/Desktop/Lung Cancer Detection/luna16_manual_labels/"

    dst_dir = "/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_manual/"

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        if "1.3.6.1.4" not in patient_id:
            continue

        print(patient_id)
        # if not "172845185165807139298420209778" in patient_id:
        #     continue
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = load_patient_images(patient_id, "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/", "*" + CUBE_IMGTYPE_SRC + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["x"] * images.shape[2])
            coord_y = int(row["y"] * images.shape[1])
            coord_z = int(row["z"] * images.shape[0])
            diameter = int(row["d"] * images.shape[2])
            node_type = int(row["id"])
            malscore = int(diameter)
            malscore = min(25, malscore)
            malscore = max(16, malscore)
            anno_index = index
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore) + "_1_" + ("pos" if node_type == 0 else "neg") + ".png", cube_img, 8, 8)
        helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_candidate_auto_images(candidate_types=[]):
    dst_dir = "/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_auto/"

    for candidate_type in candidate_types:
        for file_path in glob.glob(dst_dir + "*_" + candidate_type + ".png"):
            os.remove(file_path)

    for candidate_type in candidate_types:
        if candidate_type == "falsepos":
            src_dir = '/Users/macbook/Desktop/Lung Cancer Detection/falsepos_labels/'
        else:
            src_dir = "/Users/macbook/Desktop/Lung Cancer Detection/Extracted_Images/"

        for index, csv_file in enumerate(glob.glob(src_dir + "*_candidates_" + candidate_type + ".csv")):
            patient_id = ntpath.basename(csv_file).replace("_candidates_" + candidate_type + ".csv", "")
            print(index, ",patient: ", patient_id, " type:", candidate_type)
            # if not "148229375703208214308676934766" in patient_id:
            #     continue
            df_annos = pandas.read_csv(csv_file)
            if len(df_annos) == 0:
                continue
            images = load_patient_images(patient_id,"/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/", "*" + CUBE_IMGTYPE_SRC + ".png", exclude_wildcards=[])

            row_no = 0
            for index, row in df_annos.iterrows():
                coord_x = int(row["coord_x"] * images.shape[2])
                coord_y = int(row["coord_y"] * images.shape[1])
                coord_z = int(row["coord_z"] * images.shape[0])
                anno_index = int(row["anno_index"])
                cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 48)
                if cube_img.sum() < 10:
                    print("Skipping ", coord_x, coord_y, coord_z)
                    continue
                # print(cube_img.sum())
                try:
                    save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_0_" + candidate_type + ".png", cube_img, 6, 8)
                except Exception as ex:
                    print(ex)

                row_no += 1
                max_item = 240 if candidate_type == "white" else 200
                if candidate_type == "luna":
                    max_item = 500
                if row_no > max_item:
                    break


def load_patient_images(patient_id, base_dir=None, wildcard="*.*", exclude_wildcards=[]):
    if base_dir == None:
        base_dir = "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/*"
    src_dir = base_dir + "*" + patient_id[-5:] + "/"
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

make_annotation_images_lidc()
make_pos_annotation_images_manual()
make_candidate_auto_images(["falsepos", "luna", "edge"])



