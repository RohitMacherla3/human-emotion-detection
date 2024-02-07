from fastapi import APIRouter
from service.api.endpoints.detect import detection_router
from service.api.endpoints.test import test_router

api_router = APIRouter()
api_router.include_router(detection_router)
api_router.include_router(test_router)