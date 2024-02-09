from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
from service.core.logic.Emotion_Detector import Emotion_Detector
import numpy as np
from service.core.schemas.output import APIOutput

detection_router = APIRouter()

@detection_router.post("/DetectEmotions", response_model=APIOutput)
async def detect_emotions(im: UploadFile):
    
    if im.filename.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        pass
    else:
        raise HTTPException(status_code=415, detail='Unsupported File Format')
    
    image = Image.open(BytesIO(im.file.read()))
    image = np.array(image)
    
    return Emotion_Detector(image)
            
