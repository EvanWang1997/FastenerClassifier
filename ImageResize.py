import cv2
import os


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





