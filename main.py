import os
from pathlib import Path

import boto3
from fastapi import FastAPI
from pydantic import BaseModel
from smart_open import open

app = FastAPI()

from diarize_api import diarize


class ModelParams(BaseModel):
    audio_file: str
    model_name: str = "base.en"
    suppress_numerals: bool = False

@app.get("/")
async def root(params: ModelParams):

    session = boto3.Session(
        aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
        aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
    )

    original_file_name = Path(params.audio_file).name
    with open(params.audio_file, 'rb', transport_params={'client': session.client('s3')}) as vocal_target:
        diarize(params.model_name, vocal_target, original_file_name, params.suppress_numerals)