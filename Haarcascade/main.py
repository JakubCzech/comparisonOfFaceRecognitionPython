import cv2 as cv
import glob
from matplotlib import pyplot as plt


path = "img\*.*"
images = []
for file in glob.glob(path):
    image = cv.imread(file)
    images.append(image)

# fig=plt.figure()
# rows = len(images)
# i = 1
# for image in images:
#     fig.add_subplot(rows, 1, i)
#     color_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     plt.imshow(color_image)
#     plt.axis('off')
#     plt.title("Image "+str(i))
#     i += 1
# # plt.show()

def detectAndDisplay(image):
    face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt2.xml")
    eyes_cascade = cv.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

    frame_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    faces = face_cascade.detectMultiScale(frame_gray)
    # Average size of face on image
    num_of_faces = 0
    avg_w =0
    avg_h =0
    for (x, y, w, h) in faces:
        avg_w +=w
        avg_h +=h
    avg_w /= len(faces)
    avg_h /= len(faces)

    # Bound
    range_up = .35
    range_down= .20
    w_lowwer_bound = avg_w - range_down*avg_w
    h_lowwer_bound = avg_h - range_down*avg_h
    w_upper_bound = avg_w + range_up* avg_w
    h_upper_bound = avg_h + range_up* avg_h

    #Verification
    for (x,y,w,h) in faces:
        if(w>w_lowwer_bound and h >h_lowwer_bound and w < w_upper_bound and h <h_upper_bound):
            center_face = (x + w//2, y + h//2)
            image = cv.ellipse(image, center_face, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            num_of_faces += 1
        else:
            center_face = (x + w // 2, y + h // 2)
            image = cv.ellipse(image, center_face, (w // 2, h // 2), 0, 0, 360, (255, 0, 0), 4)


    return image , num_of_faces


fig1=plt.figure(figsize=(10,30))
for id,image in enumerate(images):
    ret_image, face_quantity = detectAndDisplay(image)
    fig1.add_subplot(len(images), 1, id+1)
    color_image = cv.cvtColor(ret_image, cv.COLOR_BGR2RGB)
    plt.imshow(color_image)
    plt.axis('off')
    plt.title("Image: "+str(id)+" Face: "+ str(face_quantity))
plt.show()



