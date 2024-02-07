from fastapi import FastAPI

app = FastAPI()

@app.get("/entry")
def read():
    return {'Hello': 'world'}
            
