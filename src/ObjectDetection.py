import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image
from zipfile import ZipFile
from urllib.request import urlretrieve

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/qhhlqcica1nvtaw/opencv_bootcamp_assets_NB1.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "opencv_bootcamp_assets_NB1.zip")

# Download if asset ZIP does not exist.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Split the image into the B,G,R components
img_bottle = cv2.imread("bottle.jpg", cv2.IMREAD_COLOR)
r, g, b = cv2.split(img_bottle)

# Show the channels
plt.figure(figsize=[20, 5])

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")

# Merge the individual channels into a BGR image
imgMerged = cv2.merge((r, g, b))

# Show the merged output
plt.subplot(144)
plt.imshow(imgMerged[:, :, ::-1])
plt.title("Merged Output")
print(img_bottle)
plt.show()
#Reverse Channel Order

#Show bottle only
cropped_bottle_bgr = imgMerged[800:2100, 1100:1700]
cropped_bottle = cropped_bottle_bgr[:,:,::-1]
plt.imshow(cropped_bottle);plt.title("Bottle Croppped")
#Using 'dsize' resizing the image while maintaining aspect ratio
desired_width = 500
aspect_ratio = desired_width / cropped_bottle.shape[1]
desired_height = int(cropped_bottle.shape[0] * aspect_ratio)
dim = (desired_width, desired_height)

# Resize image
resized_cropped_region = cv2.resize(cropped_bottle, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()
# Save resized image to disk
cv2.imwrite("resized_bottle.png", resized_cropped_region)

# Display the cropped and resized image
Image(filename="resized_bottle.png")

#Annotation in Text
imageText = img_bottle.copy()
text = "Mai Dubai Water Bottle"
fontScale = 7
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontColor = (0, 255, 0)
fontThickness = 5

cv2.putText(imageText, text, (200, 2300), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

# Display the image
plt.imshow(imageText[:, :, ::-1])
plt.show()

#Brightness of Picture
matrix = np.ones(img_bottle.shape, dtype="uint8") * 50
img_bottle = cv2.cvtColor(img_bottle, cv2.COLOR_BGR2RGB)
img_rgb_brighter = cv2.add(img_bottle, matrix)
img_rgb_darker   = cv2.subtract(img_bottle, matrix)


# Show the images
plt.figure(figsize=[18, 5])
plt.subplot(131); plt.imshow(img_rgb_darker);  plt.title("Darker");
plt.subplot(132); plt.imshow(img_bottle);         plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_brighter);plt.title("Brighter");
plt.show()

#Contrast of images using Multiplication (Overflow handling)

matrix_low_contrast = np.ones(img_bottle.shape) * 0.8
matrix_high_contast = np.ones(img_bottle.shape) * 1.2

img_rgb_lower_contrast  = np.uint8(cv2.multiply(np.float64(img_bottle.shape), matrix_low_contrast))
img_rgb_higher_contrast = np.uint8(np.clip(cv2.multiply(np.float64(img_bottle.shape), matrix_high_contast), 0, 255))

# Show the images
plt.figure(figsize=[18,5])
plt.subplot(131); plt.imshow(img_rgb_lower_contrast); plt.title("Lower Contrast");
plt.subplot(132); plt.imshow(img_bottle);       plt.title("Original");
plt.subplot(133); plt.imshow(img_rgb_higher_contrast);plt.title("Higher Contrast");
plt.show()

#Thresholding
img_thresh_adp = cv2.adaptiveThreshold(img_bottle, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)
plt.figure(figsize=[18,15])
plt.subplot(224); plt.imshow(img_thresh_adp,  cmap="gray");  plt.title("Thresholded (adaptive)");
plt.show()
