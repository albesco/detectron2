from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo 

import cv2 as cv 
import numpy as np 

class Detector:
    def __init__(self , model_type , color_type ):
        self.cfg = get_cfg()
        self.model_type = model_type
        
        if color_type == "RGB":
            self.cm = ColorMode.IMAGE
        elif color_type == "BW":
            self.cm = ColorMode.IMAGE_BW
  
        # Load model config and pre-trained model
        
        if model_type == "KP":  # Keypoint detection
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml")
        elif model_type == "PS":  # Panoptic segmentation
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        elif model_type == "OD":  # Object detection     
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")   
        elif model_type == "IS":  # Object segmentation
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "LVIS":  # LVIS instance segmentation
            self.cfg.merge_from_file( model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" # you can set "cpu" or "cuda"
        
        self.predictor = DefaultPredictor( self.cfg )
        
    def onImage(self, imagePath ):
        image = cv.imread( imagePath )
        if self.model_type != "PS": 
            predictions = self.predictor( image )
            viz = Visualizer( image[ : , : , : : -1 ] , 
                            metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0] ), 
                            instance_mode = self.cm )
            #instance_mode = ColorMode.SEGMENTATION
            output = viz.draw_instance_predictions( predictions["instances"].to("cpu"))
        else: 
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            viz = Visualizer( image[ : , : , : : -1 ] , 
                            metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0] ), 
                            )
            output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
        cv.imshow( "Result" , output.get_image()[ : , : , : : -1 ] )
        cv.waitKey(0)
    
    def onVideo(self, videoPath ):
        cap = cv.VideoCapture( videoPath )
        cont_frame = 0 
        cv.namedWindow("Result", cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow("Result", 800, 600)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            num_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
            cont_frame = cont_frame + 1
            print(f"frame {int(num_frame)} / {cont_frame}")
            if self.model_type != "PS": 
                predictions = self.predictor( frame )
                viz = Visualizer( frame[ : , : , : : -1 ] , 
                                metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0] ), 
                                instance_mode = self.cm )
                output = viz.draw_instance_predictions( predictions["instances"].to("cpu"))
            else: 
                predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
                viz = Visualizer( frame[ : , : , : : -1 ] , 
                                metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0] ), 
                                )
                output = viz.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            cv.imshow( "Result" , output.get_image()[ : , : , : : -1 ] )
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv.destroyAllWindows()
        