# Violence & Weapon Detection — Web App

## What's in here
- `app.py` — FastAPI backend, loads all 3 trained models once at startup
- `src/model.py`, `src/model_fusion.py` — model architecture definitions (needed to load the checkpoints)
- `static/index.html` — frontend (video upload + live camera modes)
- `weights/` - trained data from the google colab code
- `Dockerfile` — used by Render to build a container with ffmpeg (required for audio decoding)


## Google colab link

Go to this link below to get the code to actually train this model

https://colab.research.google.com/drive/1RxZ4GCkSw7BJD6LKiBBr9wtFO8Zrsklc?usp=sharing

## To Run 

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://localhost:8000 in your browser


