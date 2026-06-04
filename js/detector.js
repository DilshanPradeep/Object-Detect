/* ============================================================
   detector.js — YOLO11n ONNX Object Detection Engine
   Handles model loading, preprocessing, inference, and
   post-processing (confidence filtering + NMS)
   ============================================================ */

/**
 * COCO dataset 80 class names (standard ordering for YOLO models)
 */
const COCO_CLASSES = [
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
  'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
  'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
  'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
  'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
  'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
  'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
  'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
  'hair drier', 'toothbrush'
];

/**
 * Distinct color palette for bounding boxes (per-class)
 * Each color is [R, G, B]
 */
const CLASS_COLORS = [
  [79,110,247],[16,185,129],[245,158,11],[239,68,68],[139,92,246],
  [236,72,153],[6,182,212],[249,115,22],[132,204,22],[168,85,247],
  [20,184,166],[244,63,94],[99,102,241],[234,179,8],[14,165,233],
  [225,29,72],[34,197,94],[251,146,60],[167,139,250],[45,212,191],
  [248,113,113],[96,165,250],[74,222,128],[252,211,77],[129,140,248],
  [244,114,182],[34,211,238],[251,191,36],[163,230,53],[232,121,249],
  [56,189,248],[250,204,21],[45,212,191],[248,113,113],[96,165,250],
  [74,222,128],[252,211,77],[129,140,248],[244,114,182],[34,211,238],
  [79,110,247],[16,185,129],[245,158,11],[239,68,68],[139,92,246],
  [236,72,153],[6,182,212],[249,115,22],[132,204,22],[168,85,247],
  [20,184,166],[244,63,94],[99,102,241],[234,179,8],[14,165,233],
  [225,29,72],[34,197,94],[251,146,60],[167,139,250],[45,212,191],
  [248,113,113],[96,165,250],[74,222,128],[252,211,77],[129,140,248],
  [244,114,182],[34,211,238],[251,191,36],[163,230,53],[232,121,249],
  [56,189,248],[250,204,21],[45,212,191],[248,113,113],[96,165,250],
  [74,222,128],[252,211,77],[129,140,248],[244,114,182],[34,211,238]
];

/**
 * Main Object Detector class.
 * Manages the ONNX Runtime inference session and the full
 * detection pipeline: preprocess → infer → postprocess.
 */
export class ObjectDetector {
  constructor() {
    /** @type {ort.InferenceSession|null} */
    this.session = null;

    /** Model input dimensions */
    this.modelWidth = 640;
    this.modelHeight = 640;

    /** Confidence threshold (adjustable via slider) */
    this.confidenceThreshold = 0.35;

    /** IoU threshold for Non-Maximum Suppression */
    this.iouThreshold = 0.45;

    /** Number of COCO classes */
    this.numClasses = 80;

    /** Off-screen canvas for preprocessing */
    this._preprocessCanvas = document.createElement('canvas');
    this._preprocessCanvas.width = this.modelWidth;
    this._preprocessCanvas.height = this.modelHeight;
    this._preprocessCtx = this._preprocessCanvas.getContext('2d', { willReadFrequently: true });

    /** Model loaded flag */
    this.isLoaded = false;
  }

  /**
   * Initialize the ONNX Runtime session with the YOLO11n model.
   * @param {string} modelPath — Path to the .onnx file
   * @param {function} onProgress — Optional progress callback
   */
  async init(modelPath = './models/yolo11n.onnx', onProgress = null) {
    try {
      // Configure WASM backend paths
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/';

      if (onProgress) onProgress('Initializing ONNX Runtime...');

      // Create inference session with optimized settings
      this.session = await ort.InferenceSession.create(modelPath, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
        enableMemPattern: true
      });

      this.isLoaded = true;
      if (onProgress) onProgress('Model loaded successfully!');
      console.log('[Detector] YOLO11n ONNX model loaded');
      console.log('[Detector] Input names:', this.session.inputNames);
      console.log('[Detector] Output names:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('[Detector] Failed to load model:', error);
      throw new Error(`Model loading failed: ${error.message}`);
    }
  }

  /**
   * Set the confidence threshold for filtering detections.
   * @param {number} value — Threshold between 0 and 1
   */
  setConfidenceThreshold(value) {
    this.confidenceThreshold = Math.max(0.05, Math.min(0.95, value));
  }

  /**
   * Run full detection pipeline on an image source.
   * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source
   * @returns {Array} — Array of detection objects
   */
  async detect(source) {
    if (!this.isLoaded || !this.session) {
      throw new Error('Model not loaded. Call init() first.');
    }

    // Get original source dimensions
    const origWidth = source.videoWidth || source.naturalWidth || source.width;
    const origHeight = source.videoHeight || source.naturalHeight || source.height;

    if (origWidth === 0 || origHeight === 0) {
      return [];
    }

    // Step 1: Preprocess — resize and normalize to tensor
    const inputTensor = this._preprocess(source);

    // Step 2: Run inference
    const feeds = {};
    feeds[this.session.inputNames[0]] = inputTensor;
    const results = await this.session.run(feeds);

    // Step 3: Post-process — parse output, filter, NMS
    const outputTensor = results[this.session.outputNames[0]];
    const detections = this._postprocess(outputTensor, origWidth, origHeight);

    return detections;
  }

  /**
   * Preprocess the image source into a normalized [1, 3, 640, 640] Float32 tensor.
   * Uses letterboxing to maintain aspect ratio.
   * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} source
   * @returns {ort.Tensor}
   * @private
   */
  _preprocess(source) {
    const ctx = this._preprocessCtx;
    const w = this.modelWidth;
    const h = this.modelHeight;

    // Clear and draw source centered (letterbox)
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#808080'; // Gray padding
    ctx.fillRect(0, 0, w, h);

    // Calculate letterbox dimensions
    const srcW = source.videoWidth || source.naturalWidth || source.width;
    const srcH = source.videoHeight || source.naturalHeight || source.height;
    const scale = Math.min(w / srcW, h / srcH);
    const newW = Math.round(srcW * scale);
    const newH = Math.round(srcH * scale);
    const offsetX = Math.round((w - newW) / 2);
    const offsetY = Math.round((h - newH) / 2);

    // Store letterbox params for coordinate mapping
    this._letterbox = { scale, offsetX, offsetY, srcW, srcH };

    ctx.drawImage(source, offsetX, offsetY, newW, newH);

    // Extract pixel data
    const imageData = ctx.getImageData(0, 0, w, h);
    const pixels = imageData.data; // RGBA flat array

    // Create Float32Array in CHW format, normalized [0, 1]
    const totalPixels = w * h;
    const float32Data = new Float32Array(3 * totalPixels);

    for (let i = 0; i < totalPixels; i++) {
      const rgbaIdx = i * 4;
      float32Data[i] = pixels[rgbaIdx] / 255.0;                    // R channel
      float32Data[totalPixels + i] = pixels[rgbaIdx + 1] / 255.0;  // G channel
      float32Data[2 * totalPixels + i] = pixels[rgbaIdx + 2] / 255.0; // B channel
    }

    return new ort.Tensor('float32', float32Data, [1, 3, h, w]);
  }

  /**
   * Post-process the raw model output tensor.
   * Output shape: [1, 84, 8400]
   *   - 84 = 4 (bbox) + 80 (class probabilities)
   *   - 8400 = number of candidate detections
   *
   * @param {ort.Tensor} outputTensor
   * @param {number} origWidth — Original image width
   * @param {number} origHeight — Original image height
   * @returns {Array} — Filtered and NMS-applied detection objects
   * @private
   */
  _postprocess(outputTensor, origWidth, origHeight) {
    const data = outputTensor.data;
    const numDetections = 8400;
    const numChannels = 4 + this.numClasses; // 84

    const candidates = [];

    // Parse each of the 8400 candidates
    // Data layout: [1, 84, 8400] — channel-first
    // data[channel * 8400 + detection_idx]
    for (let i = 0; i < numDetections; i++) {
      // Find the best class and its score
      let maxScore = 0;
      let maxClassIdx = 0;

      for (let c = 0; c < this.numClasses; c++) {
        const score = data[(4 + c) * numDetections + i];
        if (score > maxScore) {
          maxScore = score;
          maxClassIdx = c;
        }
      }

      // Filter by confidence threshold
      if (maxScore < this.confidenceThreshold) continue;

      // Extract bounding box (center format) in model coordinates
      const cx = data[0 * numDetections + i];
      const cy = data[1 * numDetections + i];
      const bw = data[2 * numDetections + i];
      const bh = data[3 * numDetections + i];

      // Convert to corner format [x1, y1, x2, y2] in model coordinates
      let x1 = cx - bw / 2;
      let y1 = cy - bh / 2;
      let x2 = cx + bw / 2;
      let y2 = cy + bh / 2;

      // Map from letterboxed model coordinates back to original image coordinates
      const lb = this._letterbox;
      x1 = (x1 - lb.offsetX) / lb.scale;
      y1 = (y1 - lb.offsetY) / lb.scale;
      x2 = (x2 - lb.offsetX) / lb.scale;
      y2 = (y2 - lb.offsetY) / lb.scale;

      // Clamp to original image boundaries
      x1 = Math.max(0, Math.min(origWidth, x1));
      y1 = Math.max(0, Math.min(origHeight, y1));
      x2 = Math.max(0, Math.min(origWidth, x2));
      y2 = Math.max(0, Math.min(origHeight, y2));

      candidates.push({
        bbox: [x1, y1, x2, y2],
        score: maxScore,
        classIdx: maxClassIdx,
        className: COCO_CLASSES[maxClassIdx],
        color: CLASS_COLORS[maxClassIdx]
      });
    }

    // Apply Non-Maximum Suppression per class
    return this._nms(candidates);
  }

  /**
   * Greedy Non-Maximum Suppression.
   * Runs per-class to avoid suppressing different object types.
   *
   * @param {Array} candidates — Pre-filtered detection candidates
   * @returns {Array} — NMS-filtered detections
   * @private
   */
  _nms(candidates) {
    if (candidates.length === 0) return [];

    // Group candidates by class
    const classGroups = {};
    for (const det of candidates) {
      if (!classGroups[det.classIdx]) classGroups[det.classIdx] = [];
      classGroups[det.classIdx].push(det);
    }

    const results = [];

    for (const classIdx in classGroups) {
      const dets = classGroups[classIdx];

      // Sort by score descending
      dets.sort((a, b) => b.score - a.score);

      const keep = [];
      const suppressed = new Set();

      for (let i = 0; i < dets.length; i++) {
        if (suppressed.has(i)) continue;
        keep.push(dets[i]);

        for (let j = i + 1; j < dets.length; j++) {
          if (suppressed.has(j)) continue;
          if (this._iou(dets[i].bbox, dets[j].bbox) > this.iouThreshold) {
            suppressed.add(j);
          }
        }
      }

      results.push(...keep);
    }

    // Sort final results by score
    results.sort((a, b) => b.score - a.score);
    return results;
  }

  /**
   * Compute Intersection-over-Union between two bounding boxes.
   * @param {Array} boxA — [x1, y1, x2, y2]
   * @param {Array} boxB — [x1, y1, x2, y2]
   * @returns {number} — IoU value between 0 and 1
   * @private
   */
  _iou(boxA, boxB) {
    const x1 = Math.max(boxA[0], boxB[0]);
    const y1 = Math.max(boxA[1], boxB[1]);
    const x2 = Math.min(boxA[2], boxB[2]);
    const y2 = Math.min(boxA[3], boxB[3]);

    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (intersection === 0) return 0;

    const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    return intersection / (areaA + areaB - intersection);
  }

  /**
   * Clean up resources.
   */
  dispose() {
    if (this.session) {
      this.session.release();
      this.session = null;
    }
    this.isLoaded = false;
  }
}

// Export class labels and colors for use by the UI module
export { COCO_CLASSES, CLASS_COLORS };
