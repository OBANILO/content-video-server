FROM python:3.11-slim

# fonts-liberation + fonts-freefont-ttf are for the song videos (serif lyrics, gold italic artist name)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg fonts-dejavu-core fonts-dejavu fonts-liberation fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# singer cut-out model for the Shorts (weather goes behind him) - downloaded once at build time
ENV U2NET_HOME=/app/models
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

COPY main.py song.py short.py ./

RUN mkdir -p /app/outputs /app/tmp

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]
