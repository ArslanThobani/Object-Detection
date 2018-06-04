from django.shortcuts import render
from django.shortcuts import render, redirect
from .models import UploadForm,Upload
from django.http import HttpResponseRedirect
from django.urls import reverse
#from .process import result
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from .yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, \
   scale_boxes
from .yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, \
                yolo_body

def new(request):
    if(request.method=="POST"):
        if os.path.isfile("media/images/project.jpg"):
            os.remove("media/images/project.jpg")


        img = UploadForm(request.POST, request.FILES)
        if img.is_valid():
            img.save()


        return redirect('/upload')
    else:
        img = UploadForm()
    return render(request,'home_new.html',{'form':img})




# Create your views here.
def home(self):
    # if os.path.isfile("media/images/arslan.jpg"):
    #     os.remove("media/images/arslan.jpg")
    #
    # if request.method=="POST":
    #     img = UploadForm(request.POST, request.FILES)
    #     if img.is_valid():
    #         img.save()
    #         if os.path.isfile("media/project.jpg"):
    #             os.remove("media/project.jpg")
            def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):


                box_scores = box_confidence * box_class_probs

                box_classes = K.argmax(box_scores, axis=-1)
                box_class_scores = K.max(box_scores, axis=-1)

                filtering_mask = box_class_scores >= threshold

                scores = tf.boolean_mask(box_class_scores, filtering_mask)
                boxes = tf.boolean_mask(boxes, filtering_mask)
                classes = tf.boolean_mask(box_classes, filtering_mask)


                return scores, boxes, classes

            def iou(box1, box2):


                xi1 = max(box1[0], box2[0])
                yi1 = max(box1[1], box2[1])
                xi2 = min(box1[2], box2[2])
                yi2 = min(box1[3], box2[3])
                inter_area = abs(xi2 - xi1) * abs(yi2 - yi1)

                box1_area = (max(box1[0], box1[2]) - min(box1[0], box1[2])) * (
                max(box1[1], box1[3]) - min(box1[1], box1[3]))
                box2_area = (max(box2[0], box2[2]) - min(box2[0], box2[2])) * (
                max(box2[1], box2[3]) - min(box2[1], box2[3]))
                union_area = box1_area + box2_area - inter_area

                iou = inter_area / union_area

                return iou

            def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):


                max_boxes_tensor = K.variable(max_boxes,
                                              dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
                K.get_session().run(
                    tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

                # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
                nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold=0.5)

                # Use K.gather() to select only nms_indices from scores, boxes and classes
                scores = K.gather(scores, nms_indices)
                boxes = K.gather(boxes, nms_indices)
                classes = K.gather(classes, nms_indices)

                return scores, boxes, classes

            def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):



                # Retrieve outputs of the YOLO model (≈1 line)
                box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

                # Convert boxes to be ready for filtering functions
                boxes = yolo_boxes_to_corners(box_xy, box_wh)

                # Use one of the functions you've implemented to perform Score-filtering with a threshold of score_threshold (≈1 line)
                scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs,
                                                           threshold=score_threshold)

                # Scale boxes back to original image shape.
                boxes = scale_boxes(boxes, image_shape)

                # Use one of the functions you've implemented to perform Non-max suppression with a threshold of iou_threshold (≈1 line)
                scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes=max_boxes,
                                                                  iou_threshold=iou_threshold)


                return scores, boxes, classes

            def predict(sess: object, image_file: object) -> object:


                # Preprocess your image
                image, image_data = preprocess_image(image_file, model_image_size=(608, 608))

                # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
                # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
                ### START CODE HERE ### (≈ 1 line)
                out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={
                    yolo_model.input: image_data, K.learning_phase(): 0
                })


                # Print predictions info
                print('Found {} boxes for {}'.format(len(out_boxes), image_file))
                # Generate colors for drawing bounding boxes.
                colors = generate_colors(class_names)
                # Draw bounding boxes on the image file
                draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
                # Save the predicted bounding box on the image
                image.save(os.path.join("", image_file), quality=90)
                # Display the results in the notebook
                output_image = scipy.misc.imread(os.path.join("", image_file))
                imshow(output_image)

                return out_scores, out_boxes, out_classes

            #tf.reset_default_graph()
            sess = K.get_session()


            # saver.restore(sess, "/tmp/model.ckpt")
            # print("Model restored.")
            class_names = read_classes("detect/model_data/coco_classes.txt")
            anchors = read_anchors("detect/model_data/yolo_anchors.txt")
            image_shape = (720., 1280.)

            yolo_model = load_model("detect/model_data/yolo.h5")

            yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

            scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

            out_scores, out_boxes, out_classes = predict(sess, "media/images/project.jpg")
            K.clear_session()

            #return HttpResponseRedirect(reverse('/new'))
            return redirect("/result")

