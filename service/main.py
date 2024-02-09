from fastapi import FastAPI
from service.api.api import api_router
import onnxruntime as rt
import os
import uvicorn

app = FastAPI(name='Human Emotions Detection')
app.include_router(api_router)

providers = ['CPUExecutionProvider']
ViT_model = rt.InferenceSession('Model/ViT_quantized.onnx', providers=providers)

@app.get('/')
async def root():
    return 'Welcome!'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=port)
