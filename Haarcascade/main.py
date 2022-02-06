import cv2 as cv
import glob
from matplotlib import pyplot as plt
from math import sqrt
from os.path import exists
import urllib.request

# Loading images
path = "img\*.*"
images = []
for file in glob.glob(path):
    images.append(cv.imread(file))

# Show images
fig = plt.figure()
for id, image in enumerate(images):
    fig.add_subplot(len(images), 1, id + 1)
    color_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(color_image)
    plt.axis('off')
    plt.title("Image " + str(id + 1))
plt.show()


def detectAndDisplay(image):
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=15)

    # Average size of face on image
    num_of_faces = 0
    avg = 0
    for (x, y, w, h) in faces:
        avg += w
    avg /= len(faces)
    print("Average", avg)

    # std_tmp = 0
    # for (x, y, w, h) in faces:
    #     std_tmp += (w - avg)**2
    # std_dev = sqrt(std_tmp/len(faces))
    # print("Standard deviation",std_dev)

    # Bound
    range_up = .4
    range_down = .4
    lower_bound = avg - range_down * avg
    upper_bound = avg + range_up * avg

    # Verification
    for (x, y, w, h) in faces:
        if w > lower_bound and h > lower_bound and w < upper_bound and h < upper_bound:
            center_face = (x + w // 2, y + h // 2)
            image = cv.ellipse(image, center_face, (w // 2, h // 2), 0, 0, 360, (255, 0, 0, 4), 3)
            num_of_faces += 1
    return image, num_of_faces


if not exists("haarcascade_frontalface_default.xml"):
    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    urllib.request.urlretrieve(url, filename="haarcascade_frontalface_default.xml")

fig1 = plt.figure(figsize=(10, 30))
for id, image in enumerate(images):
    ret_image, face_quantity = detectAndDisplay(image)
    fig1.add_subplot(len(images), 1, id + 1)
    color_image = cv.cvtColor(ret_image, cv.COLOR_BGR2RGB)
    plt.imshow(color_image)
    plt.axis('off')
    plt.title("Image: " + str(id + 1) + " Face: " + str(face_quantity))
plt.show()
