from Detector_Module import Detector

''' Object detection call
model_type 
- "OD" Object detection
- "IS" COCO Instance segmentation
- "KP" Keypoint detection
- "LVIS" LVIS instance segmentation
color_type
- "RGB" Color
- "BW" Black and white
'''

'''
detector = Detector( model_type = "OD", color_type = "RGB") #
detector = Detector( model_type = "IS", color_type = "RGB") #
detector = Detector( model_type = "PS", color_type = "RGB") #
detector = Detector( model_type = "KP", color_type = "RGB") #
detector = Detector( model_type = "LVIS", color_type = "RGB") #
'''
detector = Detector( model_type = "PS", color_type = "RGB") #



#detector.onImage( 'F:\\Alberto\\Attivita\\UnivPM\\Test_Iniziali\\complete_detectron2\\Images\\Pecore-Pastore.jpg' )

detector.onVideo( 'F:\\Alberto\\Attivita\\UnivPM\\Test_Iniziali\\Software_Running_Elaboration\\A00000_1_2_001_L2R.mp4' )

#detector.onImage( 'F:\\Alberto\\Attivita\\UnivPM\\Test_Iniziali\\complete_detectron2\\Images\\pecore.jpg' )
