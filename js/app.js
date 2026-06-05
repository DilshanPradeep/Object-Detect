/* ============================================================
   app.js — Simplified Application
   Two modes: Camera (real-time) or Upload (single image).
   ============================================================ */

import { ObjectDetector } from './detector.js';

class App {
  constructor() {
    this.detector = new ObjectDetector();
    this.stream = null;
    this.animId = null;
    this.isDetecting = false;

    // DOM refs
    this.loader = document.getElementById('loader');
    this.loaderText = document.getElementById('loader-text');
    this.landing = document.getElementById('landing');
    this.detectionView = document.getElementById('detection-view');
    this.video = document.getElementById('video');
    this.overlay = document.getElementById('overlay');
    this.canvasWrapper = document.getElementById('canvas-wrapper');
    this.btnBack = document.getElementById('btn-back');
    this.fileInput = document.getElementById('file-input');
  }

  async init() {
    try {
      this.showLoader('Loading YOLO11n model...');
      await this.detector.init('./models/yolo11n.onnx', (msg) => {
        this.loaderText.textContent = msg;
      });
      this.hideLoader();
    } catch (err) {
      this.loaderText.textContent = 'Failed to load model. Place yolo11n.onnx in the models folder.';
      console.error(err);
      return;
    }

    this.bindEvents();
  }

  bindEvents() {
    document.getElementById('opt-camera').addEventListener('click', () => this.startCamera());
    document.getElementById('opt-upload').addEventListener('click', () => this.fileInput.click());
    this.fileInput.addEventListener('change', (e) => this.handleUpload(e));
    this.btnBack.addEventListener('click', () => this.goBack());
  }

  /* ---- Navigation ---- */

  showDetectionView() {
    this.landing.classList.add('hidden');
    this.detectionView.classList.add('active');
    this.btnBack.classList.add('visible');
  }

  goBack() {
    this.stopCamera();
    this.detectionView.classList.remove('active');
    this.landing.classList.remove('hidden');
    this.btnBack.classList.remove('visible');

    try {
      if (document.fullscreenElement || document.webkitFullscreenElement) {
        if (document.exitFullscreen) document.exitFullscreen();
        else if (document.webkitExitFullscreen) document.webkitExitFullscreen();
      }
    } catch (e) {
      console.warn("Exit fullscreen failed", e);
    }

    // Clear canvases and video
    const ctx = this.overlay.getContext('2d');
    ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    this.video.style.display = 'none';
    this.video.srcObject = null;

    // Remove any uploaded image
    const img = this.canvasWrapper.querySelector('img');
    if (img) img.remove();
  }

  /* ---- Camera Mode ---- */

  async startCamera() {
    try {
      this.showDetectionView();
      this.video.style.display = 'block';

      try {
        const el = document.documentElement;
        if (el.requestFullscreen) await el.requestFullscreen();
        else if (el.webkitRequestFullscreen) await el.webkitRequestFullscreen();
      } catch (e) {
        console.warn("Fullscreen request failed", e);
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false
      });

      this.video.srcObject = this.stream;
      await new Promise(r => { this.video.onloadedmetadata = () => { this.video.play(); r(); }; });

      this.syncOverlaySize();
      this.isDetecting = true;
      this.detectLoop();

    } catch (err) {
      console.error('Camera error:', err);
      alert('Could not access camera. Make sure you have granted permission and are using HTTPS.');
      this.goBack();
    }
  }

  stopCamera() {
    this.isDetecting = false;
    if (this.animId) { cancelAnimationFrame(this.animId); this.animId = null; }
    if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
  }

  detectLoop() {
    if (!this.isDetecting) return;

    if (this.video.readyState >= 2) {
      this.syncOverlaySize();

      this.detector.detect(this.video).then(dets => {
        if (!this.isDetecting) return;
        this.drawBoxes(dets, this.video.videoWidth, this.video.videoHeight);
        this.animId = requestAnimationFrame(() => this.detectLoop());
      }).catch(() => {
        if (this.isDetecting) this.animId = requestAnimationFrame(() => this.detectLoop());
      });
    } else {
      this.animId = requestAnimationFrame(() => this.detectLoop());
    }
  }

  /* ---- Upload Mode ---- */

  async handleUpload(e) {
    const file = e.target.files[0];
    e.target.value = '';
    if (!file || !file.type.startsWith('image/')) return;

    this.showDetectionView();
    this.video.style.display = 'none';

    // Show image
    const img = new Image();
    const url = URL.createObjectURL(file);
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = url; });

    // Remove old image if exists
    const oldImg = this.canvasWrapper.querySelector('img');
    if (oldImg) oldImg.remove();

    this.canvasWrapper.appendChild(img);

    // Wait for layout
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    this.syncOverlayToImage(img);

    this.showLoader('Detecting objects...');

    try {
      const dets = await this.detector.detect(img);
      this.hideLoader();
      this.drawBoxesOnImage(dets, img);
    } catch (err) {
      this.hideLoader();
      console.error('Detection failed:', err);
    }

    URL.revokeObjectURL(url);
  }

  /* ---- Drawing ---- */

  syncOverlaySize() {
    const rect = this.canvasWrapper.getBoundingClientRect();
    this.overlay.width = rect.width;
    this.overlay.height = rect.height;
  }

  syncOverlayToImage(img) {
    const rect = img.getBoundingClientRect();
    const wrapperRect = this.canvasWrapper.getBoundingClientRect();
    this.overlay.width = wrapperRect.width;
    this.overlay.height = wrapperRect.height;
  }

  drawBoxes(detections, sourceW, sourceH) {
    const ctx = this.overlay.getContext('2d');
    const cw = this.overlay.width;
    const ch = this.overlay.height;
    ctx.clearRect(0, 0, cw, ch);

    if (!detections.length) return;

    // Video is object-fit:contain inside wrapper
    const scale = Math.min(cw / sourceW, ch / sourceH);
    const drawW = sourceW * scale;
    const drawH = sourceH * scale;
    const offX = (cw - drawW) / 2;
    const offY = (ch - drawH) / 2;

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const dx = x1 * scale + offX;
      const dy = y1 * scale + offY;
      const dw = (x2 - x1) * scale;
      const dh = (y2 - y1) * scale;
      const [r, g, b] = det.color;

      // Box
      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(dx, dy, dw, dh);

      // Label bg
      const label = `${det.className}  ${(det.score * 100).toFixed(0)}%`;
      ctx.font = '600 12px Inter, sans-serif';
      const tw = ctx.measureText(label).width + 10;
      const th = 20;
      const ly = dy > th ? dy - th : dy;

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.roundRect(dx, ly, tw, th, 3);
      ctx.fill();

      // Label text
      ctx.fillStyle = '#fff';
      ctx.fillText(label, dx + 5, ly + 14);
    }
  }

  drawBoxesOnImage(detections, img) {
    const ctx = this.overlay.getContext('2d');
    const cw = this.overlay.width;
    const ch = this.overlay.height;
    ctx.clearRect(0, 0, cw, ch);

    if (!detections.length) return;

    // Image is object-fit:contain
    const natW = img.naturalWidth;
    const natH = img.naturalHeight;
    const imgRect = img.getBoundingClientRect();
    const wrapRect = this.canvasWrapper.getBoundingClientRect();

    // Compute rendered image size within wrapper
    const scale = Math.min(wrapRect.width / natW, wrapRect.height / natH);
    const drawW = natW * scale;
    const drawH = natH * scale;
    const offX = (wrapRect.width - drawW) / 2;
    const offY = (wrapRect.height - drawH) / 2;

    for (const det of detections) {
      const [x1, y1, x2, y2] = det.bbox;
      const dx = x1 * scale + offX;
      const dy = y1 * scale + offY;
      const dw = (x2 - x1) * scale;
      const dh = (y2 - y1) * scale;
      const [r, g, b] = det.color;

      ctx.strokeStyle = `rgb(${r},${g},${b})`;
      ctx.lineWidth = 2;
      ctx.strokeRect(dx, dy, dw, dh);

      const label = `${det.className}  ${(det.score * 100).toFixed(0)}%`;
      ctx.font = '600 12px Inter, sans-serif';
      const tw = ctx.measureText(label).width + 10;
      const th = 20;
      const ly = dy > th ? dy - th : dy;

      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.beginPath();
      ctx.roundRect(dx, ly, tw, th, 3);
      ctx.fill();

      ctx.fillStyle = '#fff';
      ctx.fillText(label, dx + 5, ly + 14);
    }
  }

  /* ---- Loader ---- */

  showLoader(msg) {
    this.loader.classList.remove('hidden');
    this.loaderText.textContent = msg;
  }

  hideLoader() {
    this.loader.classList.add('hidden');
  }
}

document.addEventListener('DOMContentLoaded', () => new App().init());
