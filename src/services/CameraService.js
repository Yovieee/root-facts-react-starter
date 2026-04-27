export class CameraService {
  constructor() {
    this.stream = null;
    this.video = null;
    this.canvas = null;
    this.cameras = [];
    this.fpsLimit = 30;
  }

  setVideoElement(videoElement) {
    this.video = videoElement;
  }

  setCanvasElement(canvasElement) {
    this.canvas = canvasElement;
  }

  async loadCameras() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      this.cameras = devices.filter((device) => device.kind === 'videoinput');
      return this.cameras;
    } catch (error) {
      console.error('Failed to load cameras:', error);
      throw error;
    }
  }

  async startCamera(selectedCameraId) {
    try {
      if (this.stream) {
        this.stopCamera();
      }

      const constraints = {
        video: selectedCameraId ? { deviceId: { exact: selectedCameraId } } : { facingMode: 'environment' }
      };

      this.stream = await navigator.mediaDevices.getUserMedia(constraints);
      if (this.video) {
        this.video.srcObject = this.stream;
        await this.video.play();
      }
      return this.stream;
    } catch (error) {
      console.error('Failed to start camera:', error);
      throw error;
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach((track) => track.stop());
      this.stream = null;
    }
    if (this.video) {
      this.video.srcObject = null;
    }
  }

  setFPS(fps) {
    this.fpsLimit = fps;
  }

  isActive() {
    return !!this.stream;
  }

  isReady() {
    return this.video && this.video.readyState >= 2;
  }
}