import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import math
import os
import cv2
import random
import scipy.ndimage
import matplotlib.pyplot as plt
from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
import glob
import pandas
from bs4 import BeautifulSoup 
import SimpleITK

# Load the scans in given folder path
def load_scan(path):
    #print(path)
    slices = [pydicom.read_file(path + '/' + s , force= True) for s in os.listdir(path) if s.endswith('.dcm')]

    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2]) #to check if there is a z direction for the pixel size
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices
#removing the outliers 


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices ])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)
    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    outside_image = image.min()
    image[image == outside_image] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):

    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float64)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, x,y= measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def get_segmented_lungs(im, plot=False):
    # Step 1: Convert into a binary image.
    binary = im < -400
    # Step 2: Remove the blobs connected to the border of the image.
    cleared = clear_border(binary)
    # Step 3: Label the image.
    label_image = label(cleared)
    # Step 4: Keep the labels with 2 largest areas.
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                       label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
    selem = disk(10) # CHANGE BACK TO 10
    binary = binary_closing(binary, selem)
    # Step 7: Fill in the small holes inside the binary mask of lungs.
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # Step 8: Superimpose the binary mask on the input image.
    get_high_vals = binary == 0
    im[get_high_vals] = -2000
    return binary


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def dilation_morphological_operation(segmented_lungs_fill):
    dil = segmented_lungs_fill.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_lungs_fill = cv2.dilate(dil,kernel,iterations = 1)
    return dilated_lungs_fill

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def process_lidc_annotations(only_patient=None, agreement_threshold=0):
    # lines.append(",".join())
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []
    count2 = 0
    for path, subdirs, files in os.walk("/Users/macbook/Desktop/LIDC-IDRI"):
        for name in subdirs:
            length = len(os.path.join(path,name))
            anno_dir = os.path.join(path,name)
            xml_paths = glob.glob(anno_dir + "/*.xml")
            for xml_path in xml_paths:
                try:
                    print(file_no, ": ",  xml_path)
                    pos, neg, extended = load_lidc_xml(anno_dir = anno_dir, xml_path=xml_path, only_patient=only_patient, agreement_threshold=agreement_threshold)
                    if pos is not None:
                        pos_count += len(pos)
                        neg_count += len(neg)
                        print("Pos: ", pos_count, " Neg: ", neg_count)
                        file_no += 1
                        all_lines += extended
                except ValueError:
                    count2 = count2 + 1
                    pass
            # if file_no > 10:
            #     break

            # extended_line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    df_annos.to_csv("lidc_annotations.csv", index=False)

def load_lidc_xml(anno_dir, xml_path, agreement_threshold=0, only_patient=None, save_nodules=False):
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text
    if only_patient is not None:
        if only_patient != patient_id:
            return None, None, None


    src_path = anno_dir



    



    print(patient_id)
    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(src_path)
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    #itk_img = load_scan(src_path)
    #img_array = get_pixels_hu(itk_img)
    rescale = spacing / 1.00

    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        # print("Sesion")
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            # print("  ", nodule.noduleID)
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = np.round(x_center / img_array.shape[2], 4)
            y_center_perc = np.round(y_center / img_array.shape[1], 4)
            z_center_perc = np.round(z_center / img_array.shape[0], 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = np.round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            z_center -= origin[2]
            z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = np.round(x_center / img_array.shape[2], 4)
            y_center_perc = np.round(y_center / img_array.shape[1], 4)
            z_center_perc = np.round(z_center / img_array.shape[0], 4)
            diameter_perc = np.round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            # print("Non nodule!", z_center)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)
            # else:
            #     print("Too few overlaps")
        pos_lines = filtered_lines

    os.chdir('/Users/macbook/Desktop/Lung Cancer Detection/Extracted_Images/')
    df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(patient_id + "_annos_pos_lidc.csv", index=False)
    df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_neg_annos.to_csv( patient_id + "_annos_neg_lidc.csv", index=False)
    process_luna_candidates_patient(src_path, patient_id)
    process_auto_candidates_patient(src_path, patient_id, sample_count=200, candidate_type="edge")

    # return [patient_id, spacing[0], spacing[1], spacing[2]]
    return pos_lines, neg_lines, extended_lines


def process_luna_candidates_patient(src_path, patient_id):
    dst_dir = '/Users/macbook/Desktop/Lung Cancer Detection/Extracted_Images/'
    img_dir = dst_dir + patient_id + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    pos_annos_manual = None
    manual_path = "/Users/macbook/Desktop/Lung Cancer Detection/luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(src_path)
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = np.array(itk_img.GetOrigin())
    spacing = np.array(itk_img.GetSpacing())
    rescale = spacing / 1.00
    direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    src_df = pandas.read_csv("/Users/macbook/Desktop/Lung Cancer Detection/candidates_V2.csv")
    src_df = src_df[src_df["seriesuid"] == patient_id]
    src_df = src_df[src_df["class"] == 0]
    patient_imgs = load_patient_images(patient_id, "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/", "*" + "_i" + ".png")
    candidate_list = []

    for df_index, candiate_row in src_df.iterrows():
        node_x = candiate_row["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = candiate_row["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = candiate_row["coordZ"]
        candidate_diameter = 6
        # print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(candidate_diameter, 2)))
        center_float = np.array([node_x, node_y, node_z])
        center_int = np.rint((center_float-origin) / spacing)
        # center_int = numpy.rint((center_float - origin) )
        # print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
        center_float_rescaled = (center_float - origin) / 1.00
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        coord_x = center_float_rescaled[0]
        coord_y = center_float_rescaled[1]
        coord_z = center_float_rescaled[2]

        ok = True

        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 64):  #  make sure we have a big margin
                ok = False
                print("################### Too close", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None and ok:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * patient_imgs.shape[2]
                pos_coord_y = row["y"] * patient_imgs.shape[1]
                pos_coord_z = row["z"] * patient_imgs.shape[0]
                diameter = row["d"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 72):  #  make sure we have a big margin
                    ok = False
                    print("################### Too close", center_float_rescaled)
                    break

        if not ok:
            continue

        candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_luna.csv", index=False)

def process_auto_candidates_patient(src_path, patient_id, sample_count=1000, candidate_type="white"):
    dst_dir = '/Users/macbook/Desktop/Lung Cancer Detection/Extracted_Images/'
    img_dir = "/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/*" + patient_id[-5:] + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")

    pos_annos_manual = None
    manual_path = "/Users/macbook/Desktop/Lung Cancer Detection/luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    reader = SimpleITK.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(src_path)
    reader.SetFileNames(dicom_names)
    itk_img = reader.Execute()
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    spacing = np.array(itk_img.GetSpacing())
    rescale = spacing / 1.00
    if candidate_type == "white":
        wildcard = "*_c.png"
    else:
        wildcard = "*_m.png"

    src_files = glob.glob(img_dir + wildcard)
    src_files.sort()
    src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]

    candidate_list = []
    tries = 0
    while len(candidate_list) < sample_count and tries < 10000:
        tries += 1
        coord_z = int(np.random.normal(len(src_files) / 2, len(src_files) / 6))
        coord_z = max(coord_z, 0)
        coord_z = min(coord_z, len(src_files) - 1)
        candidate_map = src_candidate_maps[coord_z]
        if candidate_type == "edge":
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

        non_zero_indices = np.nonzero(candidate_map)
        if len(non_zero_indices[0]) == 0:
            continue
        nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
        coord_y = non_zero_indices[0][nonzero_index]
        coord_x = non_zero_indices[1][nonzero_index]
        ok = True
        candidate_diameter = 6
        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * src_candidate_maps[0].shape[1]
            pos_coord_y = row["coord_y"] * src_candidate_maps[0].shape[0]
            pos_coord_z = row["coord_z"] * len(src_files)
            diameter = row["diameter"] * src_candidate_maps[0].shape[1]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 48): #  make sure we have a big margin
                ok = False
                print("# Too close", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * src_candidate_maps[0].shape[1]
                pos_coord_y = row["y"] * src_candidate_maps[0].shape[0]
                pos_coord_z = row["z"] * len(src_files)
                diameter = row["d"] * src_candidate_maps[0].shape[1]
                # print((pos_coord_x, pos_coord_y, pos_coord_z))
                # print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 72):  #  make sure we have a big margin
                    ok = False
                    print("#Too close",  (coord_x, coord_y, coord_z))
                    break

        if not ok:
            continue


        perc_x = np.round(coord_x / src_candidate_maps[coord_z].shape[1], 4)
        perc_y = np.round(coord_y / src_candidate_maps[coord_z].shape[0], 4)
        perc_z = np.round(coord_z / len(src_files), 4)
        candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(candidate_diameter / src_candidate_maps[coord_z].shape[1], 4), 0])

    if tries > 9999:
        print("****** WARING!! TOO MANY TRIES ************************************")
    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_" + candidate_type + ".csv", index=False)


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
    res = np.vstack(images)
    return res



process_lidc_annotations(only_patient=None, agreement_threshold=0)
print(count2)

'''count = 1
#analysing patients
OUTPUT = '/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/'
for path, subdirs, files in os.walk("/Users/macbook/Desktop/LIDC-IDRI"):
    for name in files:
        length = len(os.path.join(path,name))
        x = os.path.join(path,name)
        #print(x)
        #print(len(x))
        #y = len(x)
        #a.append(y)
        if x.endswith('.dcm'):
            print(count)
            os.chdir('/Users/macbook/Desktop/Lung Cancer Detection/OUTPUT/')
            dst_dir = x[-16:-10]
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            patient = load_scan(os.path.join(path,name[:-10]))
            patient_pixels = get_pixels_hu(patient)
            pix_resampled, spacing = resample(patient_pixels, patient, [1,1,1]) #for each slice in the shape we need a png. add the for loop here(pix_resampled.shape[0])
            for i in range(pix_resampled.shape[0]):
                img = pix_resampled[i]
                #plot_3d(pix_resampled, 400)
                #segmented_lungs = get_segmented_lungs(img.copy(), False)
                segmented_lungs_fill = get_segmented_lungs(img.copy())
                #dilated_image = dilation_morphological_operation(segmented_lungs_fill)
                #normalized_image = normalize(dilated_image)
                img = normalize(img)
                #print(pix_resampled.shape)
                cv2.imwrite(os.path.join(dst_dir, "img_" + str(i).rjust(4, '0') + "_i.png"), img * 255)
                cv2.imwrite(os.path.join(dst_dir, "img_" + str(i).rjust(4, '0') + "_m.png"), segmented_lungs_fill * 255)
            #final_preprocessed_image = zero_center(normalized_image)
            #plot_3d(dilated_lungs_fill,0)
            #plot_3d(segmented_lungs, 0)
            #plot_3d(segmented_lungs_fill, 0)
            #np.save(os.path.join(OUTPUT, 'image' + str(count)),final_preprocessed_image)
            count = count + 1
        break
''' 
#process_lidc_annotations(only_patient=None, agreement_threshold=0)
#a.sort()
#print(a)














