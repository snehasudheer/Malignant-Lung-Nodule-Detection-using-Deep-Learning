
import sys
import os
import glob
import random
import pandas
from keras.models import Sequential
import ntpath
import cv2
import numpy
from typing import List, Tuple
from keras.optimizers import SGD
from keras.layers import Input, merge, Convolution3D, MaxPooling3D, Flatten, Dense, Dropout, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
import keras.backend.tensorflow_backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math
import shutil


# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = 41
POS_WEIGHT = 2
NEGS_PER_POS = 20
P_TH = 0.6
# POS_IMG_DIR = "luna16_train_cubes_pos"
LEARN_RATE = 0.001

USE_DROPOUT = False

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def get_train_holdout_files(fold_count, train_percentage=80, logreg=True, ndsb3_holdout=0, manual_labels=True, full_luna_set=False):
    print("Get train/holdout files.")
    # pos_samples = glob.glob(settings.BASE_DIR_SSD + "luna16_train_cubes_pos/*.png")
    pos_samples = glob.glob("/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_lidc/*.png")
    print("Pos samples: ", len(pos_samples))

    pos_samples_manual = glob.glob("/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_manual/*_pos.png")
    print("Pos samples manual: ", len(pos_samples_manual))
    pos_samples += pos_samples_manual

    random.shuffle(pos_samples)
    train_pos_count = int((len(pos_samples) * train_percentage) / 100)
    pos_samples_train = pos_samples[:train_pos_count]
    pos_samples_holdout = pos_samples[train_pos_count:]
    if full_luna_set:
        pos_samples_train += pos_samples_holdout
        if manual_labels:
            pos_samples_holdout = []


    #ndsb3_list = glob.glob(settings.BASE_DIR_SSD + "generated_traindata/ndsb3_train_cubes_manual/*.png")


    if manual_labels:
        for times_ndsb3 in range(4):  # make ndsb labels count 4 times just like in LIDC when 4 doctors annotated a nodule
            pos_samples_train += pos_samples_ndsb3_fold
            pos_samples_holdout += pos_samples_ndsb3_holdout

    neg_samples_edge = glob.glob("/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_auto/*_edge.png")
    print("Edge samples: ", len(neg_samples_edge))

    # neg_samples_white = glob.glob(settings.BASE_DIR_SSD + "luna16_train_cubes_auto/*_white.png")
    neg_samples_luna = glob.glob("/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_auto/*_luna.png")
    print("Luna samples: ", len(neg_samples_luna))

    # neg_samples = neg_samples_edge + neg_samples_white
    neg_samples = neg_samples_edge + neg_samples_luna
    random.shuffle(neg_samples)

    train_neg_count = int((len(neg_samples) * train_percentage) / 100)

    neg_samples_falsepos = []
    for file_path in glob.glob("/Users/macbook/Desktop/Lung Cancer Detection/train_cubes_auto/*_falsepos.png"):
        neg_samples_falsepos.append(file_path)
    print("Falsepos LUNA count: ", len(neg_samples_falsepos))

    neg_samples_train = neg_samples[:train_neg_count]
    neg_samples_train += neg_samples_falsepos + neg_samples_falsepos + neg_samples_falsepos
    neg_samples_holdout = neg_samples[train_neg_count:]
    if full_luna_set:
        neg_samples_train += neg_samples_holdout

    train_res = []
    holdout_res = []
    sets = [(train_res, pos_samples_train, neg_samples_train), (holdout_res, pos_samples_holdout, neg_samples_holdout)]
    for set_item in sets:
        pos_idx = 0
        negs_per_pos = NEGS_PER_POS
        res = set_item[0]
        neg_samples = set_item[2]
        pos_samples = set_item[1]
        print("Pos", len(pos_samples))
        ndsb3_pos = 0
        ndsb3_neg = 0
        for index, neg_sample_path in enumerate(neg_samples):
            # res.append(sample_path + "/")
            res.append((neg_sample_path, 0, 0))
            if index % negs_per_pos == 0:
                pos_sample_path = pos_samples[pos_idx]
                file_name = ntpath.basename(pos_sample_path)
                parts = file_name.split("_")
                if parts[0].startswith("ndsb3manual"):
                    if parts[3] == "pos":
                        class_label = 1  # only take positive examples where we know there was a cancer..
                        cancer_label = int(parts[4])
                        assert cancer_label == 1
                        size_label = int(parts[5])
                        # print(parts[1], size_label)
                        assert class_label == 1
                        if size_label < 1:
                            print("huh ?")
                        assert size_label >= 1
                        ndsb3_pos += 1
                    else:
                        class_label = 0
                        size_label = 0
                        ndsb3_neg += 1
                else:
                    class_label = int(parts[-2])
                    size_label = int(parts[-3])
                    assert class_label == 1
                    assert parts[-1] == "pos.png"
                    assert size_label >= 1

                res.append((pos_sample_path, class_label, size_label))
                pos_idx += 1
                pos_idx %= len(pos_samples)

        print("ndsb2 pos: ", ndsb3_pos)
        print("ndsb2 neg: ", ndsb3_neg)

    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        class_list = []
        size_list = []
        if train_set:
            random.shuffle(record_list)
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            class_label = record_item[1]
            size_label = record_item[2]
            if class_label == 0:
                cube_image = load_cube_img(record_item[0], 6, 8, 48)
                # if train_set:
                #     # helpers.save_cube_img("c:/tmp/pre.png", cube_image, 8, 8)
                #     cube_image = random_rotate_cube_img(cube_image, 0.99, -180, 180)
                #
                # if train_set:
                #     if random.randint(0, 100) > 0.1:
                #         # cube_image = numpy.flipud(cube_image)
                #         cube_image = elastic_transform48(cube_image, 64, 8, random_state)
                wiggle = 48 - CROP_SIZE - 1
                indent_x = 0
                indent_y = 0
                indent_z = 0
                if wiggle > 0:
                    indent_x = random.randint(0, wiggle)
                    indent_y = random.randint(0, wiggle)
                    indent_z = random.randint(0, wiggle)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]

                if train_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]

                if CROP_SIZE != CUBE_SIZE:
                    cube_image = rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            else:
                cube_image = load_cube_img(record_item[0], 8, 8, 64)

                if train_set:
                    pass

                current_cube_size = cube_image.shape[0]
                indent_x = (current_cube_size - CROP_SIZE) / 2
                indent_y = (current_cube_size - CROP_SIZE) / 2
                indent_z = (current_cube_size - CROP_SIZE) / 2
                wiggle_indent = 0
                wiggle = current_cube_size - CROP_SIZE - 1
                if wiggle > (CROP_SIZE / 2):
                    wiggle_indent = CROP_SIZE / 4
                    wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1
                if train_set:
                    indent_x = wiggle_indent + random.randint(0, wiggle)
                    indent_y = wiggle_indent + random.randint(0, wiggle)
                    indent_z = wiggle_indent + random.randint(0, wiggle)

                indent_x = int(indent_x)
                indent_y = int(indent_y)
                indent_z = int(indent_z)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
                if CROP_SIZE != CUBE_SIZE:
                    cube_image = rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

                if train_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]


            means.append(cube_image.mean())
            img3d = prepare_image_for_net3D(cube_image)
            if train_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            class_list.append(class_label)
            size_list.append(size_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                y_class = numpy.vstack(class_list)
                y_size = numpy.vstack(size_list)
                yield x, {"out_class": y_class, "out_malignancy": y_size}
                img_list = []
                class_list = []
                size_list = []
                batch_idx = 0


def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False) -> Model:
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding="same")(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', padding='same', name='conv1')(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', padding='same', name='conv1a')(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), padding='valid', name='pool1')(x)

    # 2nd layer group
    x = Convolution3D(128, 3, 3, 3, activation='relu', padding='same', name='conv2a')(x)
    x = Convolution3D(128, 3, 3, 3, activation='relu', padding='same', name='conv2b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    x = Convolution3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3a')(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', padding='same', name='conv3b')(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4' )(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4a')(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv4b',)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)

    #5th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5' )(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5a')(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', padding='same', name='conv5b',)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='pool5')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)

    out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid", name="out_class_last")(x)
    out_class = Flatten(name="out_class")(out_class)

    out_malignancy = Convolution3D(1, 1, 1, 1, activation=None, name="out_malignancy_last")(x)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    model = Model(input=inputs, output=[out_class, out_malignancy])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error}, metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    if features:
        model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)

    return model


def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name, fold_count, train_full_set=False, load_weights_path=None, ndsb3_holdout=0, manual_labels=True):
    batch_size = 16
    train_files, holdout_files = get_train_holdout_files(train_percentage=80, ndsb3_holdout=ndsb3_holdout, manual_labels=manual_labels, full_luna_set=train_full_set, fold_count=fold_count)

    # train_files = train_files[:100]
    # holdout_files = train_files[:10]
    train_gen = data_generator(batch_size, train_files, True)
    holdout_gen = data_generator(batch_size, holdout_files, False)
    for i in range(0, 10):
        tmp = next(holdout_gen)
        cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
        # helpers.save_cube_img("c:/tmp/img_" + str(i) + ".png", cube_img, 4, 8)
        # print(tmp)

    learnrate_scheduler = LearningRateScheduler(step_decay)
    model = get_net(load_weight_path=load_weights_path)
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        holdout_txt = "_fs" + holdout_txt
    os.chdir("/Users/macbook/Desktop/Lung Cancer Detection/Models")
    checkpoint = ModelCheckpoint("model_" + model_name + "_" + holdout_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
    checkpoint_fixed_name = ModelCheckpoint("model_" + model_name + "_" + holdout_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #the data is input in the next line, the rest are just defining the neural network
    model.fit_generator(train_gen, len(train_files) / 1, 12, validation_data=holdout_gen, nb_val_samples=len(holdout_files) / 1, callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])

    model.save("model_" + model_name + "_" + holdout_txt + "_end.hd5")

def rescale_patient_images2(images_zyx, target_shape, verbose=False):
    if verbose:
        print("Target: ", target_shape)
        print("Shape: ", images_zyx.shape)

    # print "Resizing dim z"
    resize_x = 1.0
    interpolation = cv2.INTER_NEAREST if False else cv2.INTER_LINEAR
    res = cv2.resize(images_zyx, dsize=(target_shape[1], target_shape[0]), interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
    # print "Shape is now : ", res.shape

    res = res.swapaxes(0, 2)
    res = res.swapaxes(0, 1)

    # cv2 can handle max 512 channels..
    if res.shape[2] > 512:
        res = res.swapaxes(0, 2)
        res1 = res[:256]
        res2 = res[256:]
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res1 = cv2.resize(res1, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res2 = cv2.resize(res2, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)
        res1 = res1.swapaxes(0, 2)
        res2 = res2.swapaxes(0, 2)
        res = numpy.vstack([res1, res2])
        res = res.swapaxes(0, 2)
    else:
        res = cv2.resize(res, dsize=(target_shape[2], target_shape[1]), interpolation=interpolation)

    res = res.swapaxes(0, 2)
    res = res.swapaxes(2, 1)
    if verbose:
        print("Shape after: ", res.shape)
    return res

def load_cube_img(src_path, rows, cols, size):
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    #img is of the size 288x384, which is also 48x48x48(later split into this)
    # assert rows * size == cube_img.shape[0]
    # assert cols * size == cube_img.shape[1]
    res = numpy.zeros((rows * cols, size, size))
    img_height = size
    img_width = size

    for row in range(rows):
        for col in range(cols):
            src_y = row * img_height
            src_x = col * img_width
            res[row * cols + col] = img[src_y:src_y + img_height, src_x:src_x + img_width]
            #here the img[] has a 2d shape of 48x48
    return res

if __name__ == "__main__":
    if True:
        # model 1 on luna16 annotations. full set 1 versions for blending
        os.chdir("/Users/macbook/Desktop/Lung Cancer Detection/Models")
        train(train_full_set=True, load_weights_path=None, model_name="full", fold_count=-1, manual_labels=False)

    # model 2 on luna16 annotations + ndsb pos annotations. 3 folds (1st half, 2nd half of ndsb patients) 2 versions for blending
    '''if True:
        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=0, manual_labels=True, model_name="luna_posnegndsb_v1", fold_count=2)
        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=1, manual_labels=True, model_name="luna_posnegndsb_v1", fold_count=2)
        shutil.copy("workdir/model_luna_posnegndsb_v1__fs_h0_end.hd5", "models/model_luna_posnegndsb_v1__fs_h0_end.hd5")
        shutil.copy("workdir/model_luna_posnegndsb_v1__fs_h1_end.hd5", "models/model_luna_posnegndsb_v1__fs_h1_end.hd5")

    if True:
        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=0, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)
        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=1, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)
        shutil.copy("workdir/model_luna_posnegndsb_v2__fs_h0_end.hd5", "models/model_luna_posnegndsb_v2__fs_h0_end.hd5")
        shutil.copy("workdir/model_luna_posnegndsb_v2__fs_h1_end.hd5", "models/model_luna_posnegndsb_v2__fs_h1_end.hd5")''' 

