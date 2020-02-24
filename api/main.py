from logging import getLogger

from fastapi import FastAPI

app = FastAPI(__name__)
logger = getLogger(__name__)


@app.get('/health', status_code=200)
async def health():
    return {'status': 'alive'}


@app.post('/token', status_code=200)
async def token():
    return {'ok': True}


@app.post('/elmo', status_code=200)
async def elmo():
    return {'ok': True}


@app.post('/glove', status_code=200)
async def glove():
    return {'ok': True}
