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

      this.generator = await pipeline('text-generation', 'onnx-community/gemma-4-E2B-it-ONNX', {
        device: device,
        dtype: 'q4', // Quantized 4-bit model (~350MB) for efficiency
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
    if (!this.generator) return 'Model tidak aktif.';

    this.isGenerating = true;
    try {
      const systemInstruction =
        'You are an expert AI vegetable assistant. ' +
        'Give a response in one concise, informative sentence in Indonesian. ' +
        'Ensure the response is accurate and directly describes the requested vegetable in the specified tone.';

      const toneInstructions = {
        normal: `Berikan fakta menarik singkat tentang sayuran ${vegetableName} dengan nada santai namun informatif.`,
        funny: `Berikan fakta unik yang lucu dan menghibur tentang sayuran ${vegetableName}.`,
        professional: `Jelaskan manfaat kesehatan utama dari sayuran ${vegetableName} secara ilmiah dan profesional.`,
        casual: `Ceritakan info santai dan asyik tentang sayuran ${vegetableName} yang cocok untuk dibaca sehari-hari.`
      };

      const userInstruction = toneInstructions[this.currentTone] || toneInstructions.normal;

      const messages = [
        { role: 'system', content: systemInstruction },
        { role: 'user', content: userInstruction }
      ];

      const formattedPrompt = this.generator.tokenizer.apply_chat_template(messages, {
        tokenize: false,
        add_generation_prompt: true
      });

      // Generation Control Parameters
      const generationConfig = {
        max_new_tokens: params.max_new_tokens || 80,
        temperature: params.temperature || 0.7,
        top_p: params.top_p || 0.9,
        do_sample: params.do_sample !== undefined ? params.do_sample : true,
        return_full_text: false, // Ensure we only get the assistant's new response
      };

      const results = await this.generator(formattedPrompt, generationConfig);

      this.isGenerating = false;
      const generatedText = results[0].generated_text || '';
      return generatedText.trim();
    } catch (error) {
      this.isGenerating = false;
      console.error('Generation error:', error);
      return `Gagal menghasilkan fakta untuk ${vegetableName}.`;
    }
  }

  isReady() {
    return this.isModelLoaded;
  }
}

