import cv2
import os
import csv
import numpy as np

class ImageResizer:

    def __init__(self):
        self.datapath = "./Data/"

    """
    Resizes a single image file, creates a new image file scaled down to percent
    :param data_folder: String name of data folder 
    :param original_file_name: String name of original file
    :param conv_file_name: String name of scaled down image file
    :param percent: Num percent to scale down to for original image file
    """
    def resize_percent(self, orig_data_folder, orig_file_name, conv_data_folder, conv_file_name, percent):
        print("Running image conversion")

        # Gets original image from specified file in data
        full_file_path = orig_data_folder + orig_file_name
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)

        # Calculates converted dimensions
        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Creates converted data folder if converted data folder did not exist
        if not os.path.exists(conv_data_folder):
            os.makedirs(conv_data_folder)

        # Saves resized image to converted data folder
        full_new_file_path = conv_data_folder + conv_file_name
        cv2.imwrite(full_new_file_path, resized)

    """
    Converts all image data inside of a data folder into a new size, puts it into new folder
    :param data_folder: String name of data folder 
    :param new_folder: String name of new data folder to put the data into
    :param percent: Num percent to scale down to for each image file
    """
    def resize_all(self, data_folder, new_folder, percent):

        # Creates a new folder if the new folder name did not exist
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

        # Lops through all files and folders in the root folder
        for root, dirs, files in os.walk(data_folder):
            # Resizes all files in that specific folder
            for filename in files:
                self.resize_percent(data_folder, filename, new_folder, filename, percent)

            # Loops through all folders in this main folder
            for folder in dirs:
                if not os.path.exists(folder):
                    os.makedirs(folder)

                sub_folder = data_folder + folder
                new_sub_folder = new_folder + folder
                self.resize_all(sub_folder, new_sub_folder, percent)


    """
    Saves an image's data to a csv file
    param image_path: String name of image file to be written to the csv_file
    param csv_file: String name of csv file to be written to
    param output_class: Output type associated with entire group of images
    """
    def image_to_data(self, image_path, csv_file, output_class):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        data = np.array(img)
        flattened = data.flatten()
        final = np.append(flattened, output_class)

        with open(csv_file, 'a+') as data_file:
            writer = csv.writer(data_file)
            writer.writerow(final)

    """
    Appends a single folder's images all as data of the specified output class
    param class_folder: folder containing all images
    param csv_file: csv that is being requested to be written to
    param output_class: output class "Y" value associated with entire batch of images
    """
    def image_folder_to_data(self, class_folder, csv_file, output_class):
        for root, dirs, files in os.walk(class_folder):
            # Resizes all files in that specific folder
            with open(csv_file, 'a+') as data_file:
                writer = csv.writer(data_file)

                # Loops through all images, appends them as data into the csv with the output_class
                for image in files:
                    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
                    data = np.array(img)
                    flattened = data.flatten()
                    final = np.append(flattened, output_class)
                    writer.writerow(final)

    """
    Converts all image data to csv data
    param data_folder: Folder containing all subdata
    param csv_name: Name of csv_file to be written to
    """
    def convert_folder_classes(self, data_folder, csv_name):

        csv = open(data_folder + csv_name, 'a+')

        # Loops through all folders in the main data folder
        for root, dirs, files in os.walk(data_folder):
            # Resizes all files in that specific folder

            # Loops through all folders in this main folder
            for folder in dirs:
                folder_path_split = folder.split("/")
                class_type = folder_path_split[-1]
                self.image_folder_to_data(folder, data_folder + csv_name, class_type)










