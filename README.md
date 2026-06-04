# 🔍 Object Detection — YOLO11n Browser AI

Real-time object detection that runs **entirely in your browser** using YOLO11n ONNX and ONNX Runtime Web. No backend, no data sent to servers — 100% private.

## Features

- 🎥 Real-time webcam object detection
- 📁 Image upload detection
- 📊 Live statistics dashboard
- 📸 Screenshot capture
- 🌗 Dark/Light mode
- 📱 Mobile responsive
- ⚡ FPS counter & confidence slider

---

## Quick Start

### 1. Get the YOLO11n ONNX Model

Export from Ultralytics (requires Python):

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').export(format='onnx', opset=12, imgsz=640)"
```

Place `yolo11n.onnx` into the `models/` directory.

### 2. Run Locally (HTTP — localhost only)

```bash
npm run dev
```

Open http://localhost:8080 in your browser.

> **Note:** Webcam works on `localhost` without HTTPS. For network/mobile access, use HTTPS below.

### 3. Run with HTTPS (for mobile/network access)

#### Generate a self-signed SSL certificate:

```bash
openssl req -newkey rsa:2048 -new -nodes -x509 -days 365 -keyout key.pem -out cert.pem -subj "/CN=localhost"
```

> On Windows without OpenSSL, install it via `winget install OpenSSL` or use Git Bash which includes it.

#### Start HTTPS server:

```bash
npm run dev:https
```

#### Find your local IP:

```bash
# Windows
ipconfig
# Look for "IPv4 Address" under your WiFi adapter
```

#### Access from phone:

Open `https://<YOUR_LOCAL_IP>:8443` on your phone's browser.

> ⚠️ Your phone will show a security warning for the self-signed certificate. Tap **Advanced** → **Proceed** to continue.

---

## Project Structure

```
project/
├── index.html          # Main HTML entry point
├── css/
│   └── style.css       # Complete styling (light/dark themes)
├── js/
│   ├── app.js          # Application orchestrator
│   ├── detector.js     # ONNX inference engine
│   └── ui.js           # DOM & visual rendering
├── models/
│   └── yolo11n.onnx    # YOLO model (user-provided)
├── assets/             # Additional assets
├── package.json        # Dev server scripts
└── README.md           # This file
```

---

## Deploy to Vercel

The project includes `vercel.json` (caching, CORS headers for ONNX Runtime threading) and `.vercelignore` (excludes Python server, SSL certs, `.pt` model) for zero-config deployment.

### Option 1: GitHub + Vercel (Recommended)

1. Push this repo to GitHub (include `models/yolo11n.onnx`)
2. Go to [vercel.com](https://vercel.com) → **Add New Project**
3. Import your GitHub repo
4. Framework Preset: **Other**
5. No build command or output directory needed — it auto-detects
6. Click **Deploy** 🚀

### Option 2: Vercel CLI

```bash
npm i -g vercel
vercel
```

> **Note:** The `yolo11n.onnx` model (~10MB) is included in the deployment. Vercel's free tier supports files up to 100MB.
> Vercel provides HTTPS by default — no self-signed certs needed.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| HTML5 / CSS3 | Structure & Styling |
| Vanilla JavaScript | Application Logic |
| YOLO11n ONNX | Object Detection Model |
| ONNX Runtime Web | Browser ML Inference |
| Canvas API | Video rendering & bounding boxes |
| getUserMedia API | Webcam access |

---

## Browser Support

- ✅ Chrome 80+
- ✅ Edge 80+
- ✅ Firefox 90+
- ✅ Safari 15+
- ✅ Mobile Chrome & Safari

---

## License

MIT
