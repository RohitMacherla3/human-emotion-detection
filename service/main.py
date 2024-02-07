from fastapi import FastAPI
from service.api.api import api_router

app = FastAPI(name='Human Emotions Detection')
app.include_router(api_router)