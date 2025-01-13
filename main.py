from Detector_Module import Detector

''' Object detection call
model_type 
- "OD" Object detection
- "IS" Instance segmentation
color_type
- "RGB" Color
- "BW" Black and white
'''
detector = Detector( model_type = "IS", color_type = "RGB") #

detector.onImage( "images/pecore.jpg" )

