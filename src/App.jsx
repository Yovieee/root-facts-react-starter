import { useRef, useState, useEffect } from 'react';
import Header from './components/Header';
import CameraSection from './components/CameraSection';
import InfoPanel from './components/InfoPanel';
import { useAppState } from './hooks/useAppState';
import { CameraService } from './services/CameraService';
import { DetectionService } from './services/DetectionService';
import { RootFactsService } from './services/RootFactsService';
import { APP_CONFIG, isValidDetection } from './utils/config';
import { createDelay } from './utils/common';

function App() {
  const { state, actions } = useAppState();
  const detectionCleanupRef = useRef(null);
  const isRunningRef = useRef(false);
  const [currentTone, setCurrentTone] = useState('normal');

  // Initialize services and models
  useEffect(() => {
    const initServices = async () => {
      try {
        const camera = new CameraService();
        const detector = new DetectionService();
        const generator = new RootFactsService();

        actions.setServices({ camera, detector, generator });

        // Load detection model
        actions.setModelStatus('Memuat Detektor Sayuran (0%)...');
        await detector.loadModel((progress) => {
          actions.setModelStatus(`Memuat Detektor Sayuran (${progress}%)...`);
        });

        // Load facts generator model
        actions.setModelStatus('Memuat Otak Fakta (0%)...');
        await generator.loadModel((progress) => {
          actions.setModelStatus(`Memuat Otak Fakta (${progress}%)...`);
        });

        actions.setModelStatus('Model AI Siap');
      } catch (error) {
        console.error('Initialization failed:', error);
        actions.setError('Gagal memuat model AI. Periksa koneksi internet atau coba lagi.');
      }
    };

    initServices();

    return () => {
      if (detectionCleanupRef.current) {
        cancelAnimationFrame(detectionCleanupRef.current);
      }
    };
  }, []);

  // Detection loop
  const startDetectionLoop = async () => {
    const { detector, camera, generator } = state.services;

    if (!camera.isReady()) {
      detectionCleanupRef.current = requestAnimationFrame(startDetectionLoop);
      return;
    }

    const lastFrameTime = { current: Date.now() };
    const fpsInterval = 1000 / camera.fpsLimit;

    const loop = async () => {
      if (!isRunningRef.current) return;

      const now = Date.now();
      const elapsed = now - lastFrameTime.current;

      if (elapsed > fpsInterval) {
        lastFrameTime.current = now - (elapsed % fpsInterval);

        const result = await detector.predict(camera.video);

        if (isValidDetection(result)) {
          isRunningRef.current = false;
          actions.setRunning(false);
          actions.setAppState('analyzing');
          actions.setDetectionResult({
            className: result.label,
            score: result.confidence / 100
          });

          // Generate facts
          const fact = await generator.generateFacts(result.label);
          actions.setFunFactData(fact);
          actions.setAppState('result');
          return; // Stop loop on success
        }
      }

      detectionCleanupRef.current = requestAnimationFrame(loop);
    };

    loop();
  };

  const handleToggleCamera = async () => {
    const { camera } = state.services;
    if (state.isRunning) {
      camera.stopCamera();
      isRunningRef.current = false;
      actions.setRunning(false);
      actions.resetResults();
      if (detectionCleanupRef.current) {
        cancelAnimationFrame(detectionCleanupRef.current);
      }
    } else {
      try {
        await camera.startCamera();
        isRunningRef.current = true;
        actions.setRunning(true);
        actions.resetResults();
        await createDelay(1000);
        startDetectionLoop();
      } catch (error) {
        actions.setError('Tidak dapat mengakses kamera.');
      }
    }
  };

  const handleToneChange = (tone) => {
    setCurrentTone(tone);
    if (state.services.generator) {
      state.services.generator.setTone(tone);
    }
  };

  const handleCopyFact = async () => {
    if (state.funFactData) {
      try {
        await navigator.clipboard.writeText(state.funFactData);
        alert('Fakta menarik disalin ke clipboard!');
      } catch (err) {
        console.error('Failed to copy:', err);
      }
    }
  };

  return (
    <div className="app-container">
      <Header modelStatus={state.modelStatus} />

      <main className="main-content">
        <CameraSection
          isRunning={state.isRunning}
          onToggleCamera={handleToggleCamera}
          onToneChange={handleToneChange}
          services={state.services}
          modelStatus={state.modelStatus}
          error={state.error}
          currentTone={currentTone}
        />

        <InfoPanel
          appState={state.appState}
          detectionResult={state.detectionResult}
          funFactData={state.funFactData}
          error={state.error}
          onCopyFact={handleCopyFact}
        />
      </main>

      <footer className="footer">
        <p>Powered by TensorFlow.js & Transformers.js</p>
      </footer>

      {state.error && (
        <div style={{
          position: 'fixed',
          bottom: '1rem',
          left: '50%',
          transform: 'translateX(-50%)',
          maxWidth: '380px',
          padding: '0.875rem 1rem',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 'var(--radius-md)',
          color: '#991b1b',
          fontSize: '0.8125rem',
          boxShadow: 'var(--shadow-lg)',
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem',
          zIndex: 1000
        }}>
          <strong>Error:</strong> {state.error}
          <button
            onClick={() => actions.setError(null)}
            style={{
              marginLeft: 'auto',
              background: 'transparent',
              border: 'none',
              fontSize: '1.25rem',
              cursor: 'pointer',
              color: '#991b1b',
              padding: 0,
              lineHeight: 1
            }}
          >
            ×
          </button>
        </div>
      )}
    </div>
  );
}

export default App;

