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
detector = Detector( model_type = "PR", color_type = "RGB") #
'''

try:
    detector = Detector( model_type = "PR", color_type = "RGB") #

    detector.onImage( 'G:\\Attivita\\Progetto_Nuoto\\frame0.png' )

    #detector.onVideo( 'G:\\Attivita\\Progetto_Nuoto\\A00049_1_2_001_L2R.mp4' )
    #detector.onVideo( 'G:\\Attivita\\Progetto_Nuoto\\corsa_lenta_4m_1080p_120hz_Premiere.mp4' )

    #detector.onImage( 'F:\\Alberto\\Attivita\\UnivPM\\Test_Iniziali\\complete_detectron2\\Images\\Pecore_pastore_blu.png' )

    #detector.onImage( 'G:\\Attivita\\Progetto_Nuoto\\Pecore-Pastore_blu.png' )
except Exception as e:
    print(f"Errore: {e}")