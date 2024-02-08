from fastapi import FastAPI
from service.api.api import api_router
import onnxruntime as rt

app = FastAPI(name='Human Emotions Detection')
app.include_router(api_router)

providers = ['CPUExecutionProvider']
ViT_model = rt.InferenceSession('Model/ViT_quantized.onnx', providers=providers)

@app.get('/')
async def root():
    return 'Welcome!'