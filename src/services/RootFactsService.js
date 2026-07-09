import { pipeline, env } from '@huggingface/transformers';
import { TONE_CONFIG } from '../utils/config.js';

export class RootFactsService {
  constructor() {
    this.generator = null;
    this.isModelLoaded = false;
    this.isGenerating = false;
    this.currentTone = TONE_CONFIG.defaultTone;

    // Configure Transformers.js
    env.allowLocalModels = false; // We fetch from HF and cache via SW
    env.useBrowserCache = true;
  }

  async loadModel(onProgress) {
    try {
      // Adaptive Backend: Priority WebGPU
      const device = navigator.gpu ? 'webgpu' : 'wasm';

      this.generator = await pipeline('text2text-generation', 'nickypro/tinyllama-110M', {
        device: device,
        progress_callback: (p) => {
          if (onProgress && p.status === 'progress') {
            onProgress(Math.floor(p.progress));
          }
        }
      });

      this.isModelLoaded = true;
      return { device };
    } catch (error) {
      console.error('Failed to load facts model:', error);
      throw error;
    }
  }

  setTone(tone) {
    this.currentTone = tone;
  }

  async generateFacts(vegetableName, params = {}) {
    if (!this.generator) return 'Model not loaded.';

    this.isGenerating = true;
    try {
      const tonePrompts = {
        normal: `describe vegetable ${vegetableName} in normal way with one sentences`,
        funny: `describe vegetable ${vegetableName} in funny way with one sentences`,
        professional: `describe health benefits of vegetable ${vegetableName} in professional way with one sentences`,
        casual: `describe vegetable ${vegetableName} in casual way with one sentences`
      };

      const prompt = tonePrompts[this.currentTone] || tonePrompts.normal;

      // Generation Control Parameters
      const generationConfig = {
        max_new_tokens: params.max_new_tokens || 50,
        temperature: params.temperature || 0.7,
        top_p: params.top_p || 0.9,
        do_sample: params.do_sample !== undefined ? params.do_sample : true,
      };

      const results = await this.generator(prompt, generationConfig);

      this.isGenerating = false;
      return results[0].generated_text;
    } catch (error) {
      this.isGenerating = false;
      console.error('Generation error:', error);
      return `Failed to generate facts for ${vegetableName}.`;
    }
  }

  isReady() {
    return this.isModelLoaded;
  }
}

