import re
from model_loader import load_facenet, load_retinaface
from engine import annotate_video, extract_anchor_images_embeddings

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="label faces in video")
    
    parser.add_argument("--facenet_weights", 
                        help="file path to facenet weights", 
                        default="model_weights/facenet.h5",
                        type=str)

    parser.add_argument("--retinaface_weights",
                        help="File path to retainface weights",
                        default="model_weights/retinaface.h5",
                        type=str
                        )

    parser.add_argument("--anchor_image_dir",
                        help="filepath to the root directory of the anchor images",
                        default="./test_images",
                        type=str
                        )

    parser.add_argument("--video_fp",
                        help="filepath to video to mark face, only accepts mp4",
                        type=str,
                        required=True
                        )

    parser.add_argument("--output_fp",
                        help="file path to save the annotated video need to end with mp4",
                        type=str,
                        required=True)

    parser.add_argument("--sample_rate",
                        help='sample the frames instead of predicting on every frame',
                        default=1,
                        type=int) 

    parser.add_argument("--threshold",
                        help='cosine sim threshold to assign face to a class',
                        default=0.52,
                        type=float)

    args = parser.parse_args()
    return args



def main():
    args =parse_args()
    
    embedding_model= load_facenet(args.facenet_weights)
    detection_model = load_retinaface(args.retinaface_weights)

    reference_labels, reference_embeddings = extract_anchor_images_embeddings(embedding_model, 
                                                                              detection_model ,
                                                                              root_dir=args.anchor_image_dir)

    annotate_video(args.video_fp, reference_embeddings, 
                   reference_labels, detection_model, 
                   embedding_model, args.output_fp, 
                   args.sample_rate, args.threshold)




if __name__ == "__main__":
    main()


