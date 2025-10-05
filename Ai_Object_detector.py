
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub


detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

path = "/content/images (3).jpeg"
image = cv2.imread(path)

rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

im_tensor = tf.convert_to_tensor(rgb_image)
dim_image = tf.expand_dims(im_tensor,0)

objects = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

detected_image =detector(dim_image)

color = np.random.uniform(0,255, size=(len(objects),3))

boxes = detected_image["detection_boxes"][0].numpy()
classes = detected_image["detection_classes"][0].numpy().astype(int)
scores = detected_image["detection_scores"][0].numpy()


height, width,_ = image.shape

for i in range(len(boxes)):
        if scores[i] > 0.6:
           y1,x1,y2,x2 = boxes[i]
           x1,y1,x2,y2 = int(x1*width),int(y1*height),int(x2*width),int(y2*height)
           cv2.rectangle(image, (x1,y1),(x2,y2),color[i],2)
           label = objects.get(classes[i],"N/A")
           cv2.putText(image,f"{label}:{scores[i]:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()