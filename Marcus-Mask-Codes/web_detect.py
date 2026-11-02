import gradio as gr
import imutils
import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# load our serialized face detector model from disk
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
print("[INFO] loading face detector model... %s %s"%(prototxtPath, weightsPath))
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model... %s"%"mask_detector.model")
maskNet = load_model("mask_detector.model")

# set confidence 
min_confidence = 0.5

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    #print("detections: %s"%detections)
    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        #print("confidence:%s"%confidence )
        if confidence > min_confidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #print("i:%s, startX, startY, endX, endY :%s,%s,%s,%s "%(i, startX, startY, endX, endY))
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        for i in range(0,len(faces)):
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
            preds.append(maskNet.predict(faces[i])[0])
        #print("%s faces found, %s preds "%(len(faces),len(preds)))
        #print("preds\n%s"%(preds))
        #if len(faces) == 2 :
        #	input("enter to continue...")
        

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def predict(img):
    # ... implement face segmentation model on input 200x200 numpy array
    # ... return segmentation mask as numpy array
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    
    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    #print("locs %f"%len(locs))
    #for (i, (box, pred)) in enumerate(zip(locs, preds)):
    for i in range (0, len(preds)):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = locs[i]
        #print("%d: %f,%f,%f,%f"%(i, startX, startY, endX, endY))
        (mask, withoutMask) = preds[i]

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),color, 2)
        cv2.putText(frame, label, (startX, y),
              cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
    return frame

webcam = gr.inputs.Webcam(shape=(400, 400))
gr.Interface(fn=predict, inputs=webcam, outputs="image",capture_session=True,).launch(inbrowser=True)