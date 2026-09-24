FROM python:3.11-slim

# fonts-liberation + fonts-freefont-ttf are for the song videos (serif lyrics, gold italic artist name)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core fonts-dejavu fonts-liberation fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py song.py ./

RUN mkdir -p /app/outputs /app/tmp

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
