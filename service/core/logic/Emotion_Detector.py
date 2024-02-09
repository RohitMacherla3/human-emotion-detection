import onnxruntime as rt
import numpy as np
import cv2
import time
import service.main as model

def Emotion_Detector(image_array):
    
    im = cv2.resize(image_array, (224, 224))
    if im.shape[-1] > 3:
        im = im[..., :3]
    if im.shape[-1] == 1 or len(im.shape) ==2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    im = np.float32(im)
    im = np.expand_dims(im, axis=0)
    
    time_start = time.time()

    prediction = model.ViT_model.run(['dense'], {'input': np.array(im)})
    emotion_class = np.argmax(prediction, axis =-1)[0]
    print(emotion_class)
    
    time_elapsed = time.time() - time_start
    
    if int(emotion_class) == 0:
        emotion = 'Angry'
    elif int(emotion_class) == 1:
        emotion = 'Happy'
    elif int(emotion_class) == 2:
        emotion = 'Sad'
    
    return {'Emotion':  emotion, 
            'TimeElapsed': str(time_elapsed)}
