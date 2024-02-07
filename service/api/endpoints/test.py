from fastapi import APIRouter

test_router = APIRouter()

@test_router.get('/ServerStatus')
def Server_Running_Status():
    return 'The API Server is running'