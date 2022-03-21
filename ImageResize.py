import cv2

class ImageResizer:

    def __init__(self):
        self.datapath = "./Data/"

    def resize_percent(self, original_file_name, conv_file_name, percent):
        print("Running image conversion")

        # Gets original image from specified file in data
        full_file_path = self.datapath + original_file_name
        img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)

        # Prints original file dimensions
        print('Original Dimensions : ', img.shape)

        # Calculates converted dimensions
        width = int(img.shape[1] * percent / 100)
        height = int(img.shape[0] * percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        print('Resized Dimensions : ', resized.shape)

        full_new_file_path = self.datapath + conv_file_name

        cv2.imshow("Resized image", resized)
        cv2.waitKey(0)
        cv2.imwrite(full_new_file_path, resized)
        cv2.destroyAllWindows()


