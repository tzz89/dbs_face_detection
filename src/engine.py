import os

# object detection and recoginition library
from deepface import DeepFace
from deepface.commons import functions as deepface_functions
from retinaface import RetinaFace
from retinaface.model.retinaface_model import build_model as retinaface_build_model
import retinaface.commons.postprocess as retinaface_postprocess
from tensorflow.keras.preprocessing import image

# python libaries
import glob
from typing import Union, Dict, List
import collections

#sklearn and numeric libraries 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#opencv library
import cv2


def load_category_image_filepaths(root_dir:str="test_images")->Dict[str, List[str]]:
    """
    Given the root directory of images;
    1. read all directory names
    2. collect all image path (jpeg, png) within the directory
    """
    directories = os.listdir("test_images")
    category_image_filepaths= collections.defaultdict(list)
    
    for directory in directories:
        directory_filepath=os.path.join(root_dir, directory)
        for file in os.listdir(directory_filepath):
            full_filepath = os.path.join(directory_filepath, file)
            category_image_filepaths[directory].append(full_filepath)
            
    return dict(category_image_filepaths)     

def extract_anchor_images_embeddings(embedding_model, detection_model,root_dir:str="test_images"):
    """
    Generates all the anchor embeddings from the root folder
    """
    category_image_filepaths = load_category_image_filepaths(root_dir)
    labels = []
    anchor_embeddings = []

    for category, image_filepaths in category_image_filepaths.items():
        print(f"Category: {category}")
        for image_filepath in image_filepaths:
            print(f"Image :{image_filepath}")
            
            img = cv2.imread(image_filepath)
            _, embeddings = get_faces_embeddings(img, embedding_model, detection_model)
            
            if len(embeddings) > 1:
                print("Skipping as more than 1 face found in image")
                continue
            if len(embeddings) == 0:
                print("Skipping as no faces found in image")
                continue
            
            labels.append(category)
            anchor_embeddings.append(embeddings[0])
            
    return labels, np.array(anchor_embeddings)   

def annotate_video(video_fp:str, 
                   reference_embeddings, 
                   reference_labels:list, 
                   detection_model,
                   embedding_model,
                   output_video_fp:str)->None:
    """
    Takes in an video filepath
    For each video frame, extracts all the faces in the frame a compare with reference embeddings
    Draw BBOX and labels on the video frame and append to video recorder
    Save the video to output path
    """
    
    cap = cv2.VideoCapture(video_fp)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    
    out = cv2.VideoWriter(output_video_fp, cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width,frame_height))
    
    try:
        while True:
            ret, frame = cap.read()

            if ret == True:
                coords, embeddings = get_faces_embeddings(frame, embedding_model, detection_model)
                
                if len(embeddings)>0:
                    cos_sims = cosine_similarity(embeddings,reference_embeddings)

                    preds = np.argmax(cos_sims, axis=1)
                    # draw rectangle and predictions
                    for index, (coord, pred) in enumerate(zip(coords, preds)):      
                        frame = cv2.rectangle(frame, coord[:2], coord[2:] , color=(255, 0, 0), thickness=2)
                        print(f"cos_sims {cos_sims[index][pred]}")
                        
                        text = "UNKNOWN"
                        if cos_sims[index][pred] > 0.4:
                            text = reference_labels[pred]
                            
                        cv2.putText(frame, text, 
                                    coord[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (255, 0, 0), 2,cv2.LINE_AA)


                # display frame
                cv2.imshow("frame", frame)
        
                #save frame
                out.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    except Exception as e:
        print(e)
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def get_faces_embeddings(frame:np.ndarray, embedding_model, detector_backend):
    """
    process a image and returns the bbox coordinates and embeddings
    """
    detection_results = RetinaFace.detect_faces(frame, model=detector_backend)
    coordinates = []
    embeddings = []
    
                
    if type(detection_results) == dict: # source code shows that empty detection will return tuple
        for detection_result in detection_results.values():
            x1, y1, x2, y2 = detection_result["facial_area"]

            # alignment of face
            detected_face = frame[y1:y2, x1:x2].copy()
            left_eye = detection_result['landmarks']['left_eye']
            right_eye = detection_result['landmarks']['right_eye']
            nose = detection_result['landmarks']['nose']
            detected_face = retinaface_postprocess.alignment_procedure(detected_face, right_eye, left_eye, nose)


            # resizing to expected shape maintence aspect ratio and pad
            target_size = deepface_functions.find_input_shape(embedding_model)

            if detected_face.shape[0] > 0 and detected_face.shape[1] > 0:
                factor_0 = target_size[0] / detected_face.shape[0]
                factor_1 = target_size[1] / detected_face.shape[1]
                factor = min(factor_0, factor_1)

                dsize = (int(detected_face.shape[1] * factor), int(detected_face.shape[0] * factor))
                detected_face = cv2.resize(detected_face, dsize)

                diff_0 = target_size[0] - detected_face.shape[0]
                diff_1 = target_size[1] - detected_face.shape[1]

                detected_face = np.pad(detected_face, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

                if detected_face.shape[0:2] != target_size:
                    detected_face = cv2.resize(detected_face, target_size)

                detected_face = image.img_to_array(detected_face) #what this line doing? must?
                detected_face = np.expand_dims(detected_face, axis = 0)
                detected_face /= 255 #normalize input in [0, 1]

            # normalization
            detected_face = deepface_functions.normalize_input(detected_face, normalization='Facenet')

            embedding = embedding_model.predict(detected_face)[0]
            coordinates.append((x1, y1, x2, y2))
            embeddings.append(embedding)
            
    return coordinates, embeddings