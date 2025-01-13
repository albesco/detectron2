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
        
        if color_type == "RGB":
            self.cm = ColorMode.IMAGE
        elif color_type == "BW":
            self.cm = ColorMode.IMAGE_BW
  
        # Load model config and pre-trained model
        if model_type == "OD" :     # Object detection     
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")   
        elif model_type == "IS":    # Object segmentation
            self.cfg.merge_from_file( model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu" # you can set "cpu" or "cuda"
        
        self.predictor = DefaultPredictor( self.cfg )
        
    def onImage(self, imagePath ):
        image = cv.imread( imagePath )
        predictions = self.predictor( image )
        

            
        viz = Visualizer( image[ : , : , : : -1 ] , 
                          metadata = MetadataCatalog.get( self.cfg.DATASETS.TRAIN[0] ), 
                          instance_mode = self.cm )
        instance_mode = ColorMode.SEGMENTATION
        output = viz.draw_instance_predictions( predictions["instances"].to("cpu"))
        cv.imshow( "Result" , output.get_image()[ : , : , : : -1 ] )
        cv.waitKey(0)
        