import urllib.request
from os.path import exists
from glob import glob
from matplotlib import pyplot as plt
import cv2 as cv


# Loading images
path = r"img\*.*"
images = []
for file in glob(path):
    images.append(cv.imread(file))

# Show images
fig = plt.figure()
for image_id, image in enumerate(images):
    fig.add_subplot(len(images), 1, image_id + 1)
    color_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(color_image)
    plt.axis('off')
    plt.title("Image " + str(image_id + 1))
plt.show()


def detect_face(image_to_detect):
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    frame_gray = cv.cvtColor(image_to_detect, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=15)

    # Average size of face on image
    avg = 0
    for (x, y, w, h) in faces:
        avg += w
    avg /= len(faces)
    print(f'Average {avg}')

    # Bound
    size_range = .4
    lower_bound = avg - size_range * avg
    upper_bound = avg + size_range * avg
    num_of_faces = 0

    # Verification
    for (x, y, w, h) in faces:
        if lower_bound < w < upper_bound:
            center_face = (x + w // 2, y + h // 2)
            ret_image = cv.ellipse(image_to_detect, center_face, (w // 2, h // 2), 0, 0, 360, (255, 0, 0, 4), 3)
            num_of_faces += 1
    return ret_image, num_of_faces


if not exists("haarcascade_frontalface_default.xml"):
    url = ('https://raw.githubusercontent.com/opencv/'
           'opencv/master/data/haarcascades/'
           'haarcascade_frontalface_default.xml')
    urllib.request.urlretrieve(url, filename="haarcascade_frontalface_default.xml")

if __name__ == '__main__':
    fig1 = plt.figure(figsize=(10, 30))
    for image_id, image in enumerate(images):
        ret_image, face_quantity = detect_face(image)
        fig1.add_subplot(len(images), 1, image_id + 1)
        color_image = cv.cvtColor(ret_image, cv.COLOR_BGR2RGB)
        plt.imshow(color_image)
        plt.axis('off')
        plt.title(f'Image: {image_id+1} faces: {face_quantity}')
    plt.show()
