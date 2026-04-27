import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

export class DetectionService {
  constructor() {
    this.model = null;
    this.labels = [];
    this.config = null;
    this.currentBackend = 'cpu';
  }

  async loadModel(onProgress) {
    try {
      // Adaptive Backend Logic
      if (navigator.gpu) {
        try {
          await tf.setBackend('webgpu');
          this.currentBackend = 'webgpu';
        } catch (e) {
          console.warn('WebGPU failed, falling back to WebGL', e);
          await tf.setBackend('webgl');
          this.currentBackend = 'webgl';
        }
      } else {
        await tf.setBackend('webgl');
        this.currentBackend = 'webgl';
      }
      await tf.ready();
      console.log(`Using backend: ${tf.getBackend()}`);

      // Load Metadata
      const metadataResponse = await fetch('/model/metadata.json');
      const metadata = await metadataResponse.json();
      this.labels = metadata.labels;

      // Load Model
      this.model = await tf.loadLayersModel('/model/model.json', {
        onProgress: (p) => {
          if (onProgress) onProgress(Math.floor(p * 100));
        }
      });

      return { backend: this.currentBackend, labels: this.labels };
    } catch (error) {
      console.error('Failed to load detection model:', error);
      throw error;
    }
  }

  async predict(imageElement) {
    if (!this.model) return null;

    return tf.tidy(() => {
      // Teachable Machine image models expect 224x224 input
      const img = tf.browser.fromPixels(imageElement)
        .resizeNearestNeighbor([224, 224])
        .toFloat()
        .expandDims();

      // Normalize if needed? Usually TM models handle normalization internally or expect 0-255 or -1 to 1.
      // TM models usually use (img / 127.5) - 1.0 but some use 0-1.
      // Let's check common TM behavior. Actually most @teachablemachine/image models handle it.
      // But we are using raw tfjs.
      const normalizedImg = img.div(tf.scalar(127.5)).sub(tf.scalar(1.0));

      const predictions = this.model.predict(normalizedImg);
      const data = predictions.dataSync();

      let maxConfidence = -1;
      let maxIndex = -1;

      for (let i = 0; i < data.length; i++) {
        if (data[i] > maxConfidence) {
          maxConfidence = data[i];
          maxIndex = i;
        }
      }

      return {
        label: this.labels[maxIndex],
        confidence: Math.floor(maxConfidence * 100),
        isValid: maxConfidence > 0.01 // Minimal threshold for consideration
      };
    });
  }

  isLoaded() {
    return !!this.model;
  }
}
