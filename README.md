# Content Video Server

This server works with your WordPress DynamicTree / Content Video System snippet.

It creates tutorial, marketing, and review videos using ElevenLabs voiceover, Pexels clips, captions, and FFmpeg rendering.

## Render setup

1. Upload this folder to GitHub as a new repository.
2. Go Render → New → Web Service.
3. Connect the GitHub repository.
4. Choose Docker.
5. Add environment variable:

PUBLIC_BASE_URL=https://YOUR-RENDER-SERVICE.onrender.com

6. Deploy.

## Endpoints

POST /generate

GET /status/{api_key}

GET /outputs/{video}.mp4

## WordPress snippet URL

Replace this in your Content Video System snippet:

https://content-video-server.onrender.com/generate

with your real Render URL, for example:

https://your-service-name.onrender.com/generate

Also replace:

https://content-video-server.onrender.com/status/

with:

https://your-service-name.onrender.com/status/
