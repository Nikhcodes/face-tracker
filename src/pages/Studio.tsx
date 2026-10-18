import {
  AlertTriangle,
  Camera,
  HelpCircle,
  ListChecks,
  RefreshCw,
  SlidersHorizontal,
  SwitchCamera,
  Timer,
  Video,
  VideoOff,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react';
import { useCallback, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from 'react';
import { CAMERA_ERROR_COPY } from '../camera/camera';
import { CaptureDialog } from '../components/studio/CaptureDialog';
import { EffectsPanel } from '../components/studio/EffectsPanel';
import { PoseGuide } from '../components/studio/PoseGuide';
import { SoundPanel, setSoundEnabled } from '../components/studio/SoundPanel';
import { StatusChip, describeDisplay } from '../components/studio/StatusChip';
import { Wordmark } from '../components/Wordmark';
import { PRIVACY_STATEMENT } from '../config/copy';
import { StudioEngine, type CapturedPhoto, type StudioStatus } from '../engine/StudioEngine';
import { Link, navigate } from '../lib/router';
import { useStore } from '../lib/useStore';
import { settingsStore, updateSettings } from '../state/settings';
import { pushToast } from '../state/toasts';
import { sound } from '../audio/sound';

type Tab = 'poses' | 'effects' | 'sound';
const noopSubscribe = () => () => {};
const OFF_STATUS: StudioStatus | null = null;

function useEngineStatus(engine: StudioEngine | null): StudioStatus | null {
  return useSyncExternalStore(
    engine ? engine.status.subscribe : noopSubscribe,
    () => (engine ? engine.status.get() : OFF_STATUS),
    () => OFF_STATUS,
  );
}

/** Fits the stage frame into the available space while keeping the camera's aspect ratio. */
function useFitBox(container: HTMLElement | null, aspect: number) {
  const [box, setBox] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    if (!container) return;
    const fit = () => {
      const r = container.getBoundingClientRect();
      let width = r.width;
      let height = width / aspect;
      if (height > r.height) {
        height = r.height;
        width = height * aspect;
      }
      setBox({ width: Math.floor(width), height: Math.floor(height) });
    };
    fit();
    const ro = new ResizeObserver(fit);
    ro.observe(container);
    return () => ro.disconnect();
  }, [container, aspect]);
  return box;
}

export function Studio() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stageEl, setStageEl] = useState<HTMLDivElement | null>(null);
  const [engine, setEngine] = useState<StudioEngine | null>(null);
  const [photo, setPhoto] = useState<CapturedPhoto | null>(null);
  const [panelOpen, setPanelOpen] = useState(() => typeof window !== 'undefined' && window.matchMedia?.('(min-width: 1100px)').matches);
  const [tab, setTab] = useState<Tab>('poses');
  const [aspect, setAspect] = useState(16 / 9);
  const status = useEngineStatus(engine);
  const settings = useStore(settingsStore);
  const box = useFitBox(stageEl, aspect);
  const shutterRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    document.title = 'Camera — POSE CAM';
    const e = new StudioEngine();
    e.attach(videoRef.current!, canvasRef.current!);
    e.onToast = (t) => pushToast(t);
    e.onTimerCapture = (p) => setPhoto(p);
    setEngine(e);
    if (import.meta.env.DEV) {
      // dev-only hook for visual testing with synthetic hands (stripped from production)
      void import('../gestures/synthetic').then((synthetic) => {
        (window as unknown as { __posecam?: unknown }).__posecam = { engine: e, synthetic };
      });
    }
    // Only start automatically when the visitor just clicked "Enter the camera".
    const autostart = (window.history.state as { autostart?: boolean } | null)?.autostart === true;
    if (autostart) {
      window.history.replaceState({ autostart: false }, '');
      void e.start();
    }
    return () => {
      e.dispose();
      sound.stopMusic();
    };
  }, []);

  // keep the stage aspect in sync with the camera
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const onSize = () => v.videoWidth && setAspect(v.videoWidth / v.videoHeight);
    v.addEventListener('loadedmetadata', onSize);
    v.addEventListener('resize', onSize);
    return () => {
      v.removeEventListener('loadedmetadata', onSize);
      v.removeEventListener('resize', onSize);
    };
  }, []);

  const camera = status?.camera ?? 'off';
  const isOn = camera === 'on';

  const capture = useCallback(async () => {
    if (!engine) return;
    const p = await engine.capture();
    if (p) setPhoto(p);
  }, [engine]);

  const closePhoto = useCallback(() => {
    setPhoto((p) => {
      if (p) URL.revokeObjectURL(p.url);
      return null;
    });
    requestAnimationFrame(() => shutterRef.current?.focus());
  }, []);

  // keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (photo || e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.closest('input, select, textarea, [role="radiogroup"]') || target.isContentEditable)) return;
      if (e.key === ' ' && isOn && !(target instanceof HTMLButtonElement)) {
        e.preventDefault();
        void capture();
      } else if (e.key.toLowerCase() === 't' && isOn) engine?.startCountdown();
      else if (e.key.toLowerCase() === 'd') updateSettings({ debug: !settingsStore.get().debug });
      else if (e.key.toLowerCase() === 'm') updateSettings({ mirror: !settingsStore.get().mirror });
      else if (e.key === 'Escape' && panelOpen && !window.matchMedia('(min-width: 1100px)').matches) setPanelOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [engine, isOn, capture, photo, panelOpen]);

  const openTab = (t: Tab) => {
    if (panelOpen && tab === t) setPanelOpen(false);
    else {
      setTab(t);
      setPanelOpen(true);
    }
  };

  const display = status?.display ?? { kind: 'none' as const };
  const announce = isOn && status?.model === 'ready' ? describeDisplay(display).text : '';

  return (
    <div className="studio">
      <div className="studio__main">
        <header className="studio__top">
          <button type="button" className="icon-btn" aria-label="Exit camera" onClick={() => navigate('/')}>
            <X />
          </button>
          <Wordmark />
          <button
            type="button"
            className="icon-btn"
            aria-label={settings.sound ? 'Mute sound' : 'Turn sound on'}
            aria-pressed={settings.sound}
            onClick={() => void setSoundEnabled(!settings.sound)}
          >
            {settings.sound ? <Volume2 /> : <VolumeX />}
          </button>
          <Link to="/help" className="icon-btn" aria-label="Help">
            <HelpCircle />
          </Link>
        </header>

        <div className="stage" ref={setStageEl}>
          <div
            className="stage__frame"
            style={isOn && box.width ? { width: box.width, height: box.height } : { width: '100%', height: '100%' }}
          >
            <video ref={videoRef} className="stage__video" playsInline muted aria-hidden="true" />
            <canvas
              ref={canvasRef}
              className="stage__canvas"
              role="img"
              aria-label={isOn ? `Live camera preview. ${announce}` : 'Camera preview (off)'}
            />
            {isOn && status?.model === 'loading' ? (
              <div className="stage__banner" role="status">
                <span className="spinner mini-spinner" aria-hidden="true" />
                Loading hand tracking…
              </div>
            ) : null}
            {isOn && status?.model === 'error' ? (
              <div className="stage__banner stage__banner--error" role="alert">
                <AlertTriangle size={16} aria-hidden="true" />
                <span>Hand tracking couldn’t load. Effects and photos still work.</span>
                <button type="button" className="btn btn--sm" onClick={() => engine?.retryModel()}>
                  Retry
                </button>
              </div>
            ) : null}
            {isOn && settings.debug && status ? (
              <div className="debug" aria-hidden="true">
                {`fps ${status.stats.fps}  detect ${status.stats.detectFps}/s  ${status.stats.inferenceMs}ms\nbackend ${status.backend ?? '–'}\nframe ${status.stats.verdict || '–'}\nshown ${describeDisplay(display).text}`}
              </div>
            ) : null}
            {!isOn ? <StageOverlay status={status} onStart={() => void engine?.start()} /> : null}
          </div>
        </div>

        <div className="studio__status">
          {isOn ? <StatusChip display={display} countdown={status?.countdown ?? null} modelReady={status?.model === 'ready'} /> : null}
        </div>

        <div className="dock">
          <div className="dock__side">
            <button
              type="button"
              className="icon-btn"
              aria-label={isOn ? 'Turn camera off' : 'Turn camera on'}
              aria-pressed={isOn}
              disabled={camera === 'starting' || !engine}
              onClick={() => (isOn ? engine?.stop() : void engine?.start())}
            >
              {isOn ? <Video /> : <VideoOff />}
            </button>
            <button type="button" className="icon-btn" aria-label="Pose guide" aria-expanded={panelOpen && tab === 'poses'} onClick={() => openTab('poses')}>
              <ListChecks />
            </button>
          </div>
          <div className="studio__live">
            <button
              ref={shutterRef}
              type="button"
              className="shutter"
              aria-label="Take photo"
              disabled={!isOn}
              onClick={() => void capture()}
            />
          </div>
          <div className="dock__side dock__side--end">
            <button
              type="button"
              className="icon-btn"
              aria-label={status?.countdown != null ? 'Cancel photo timer' : 'Start 3-second photo timer'}
              aria-pressed={status?.countdown != null}
              disabled={!isOn}
              onClick={() => (status?.countdown != null ? engine?.cancelCountdown() : engine?.startCountdown())}
            >
              <Timer />
            </button>
            {(status?.devices.length ?? 0) > 1 ? (
              <button type="button" className="icon-btn" aria-label="Switch camera" disabled={!isOn} onClick={() => void engine?.switchCamera()}>
                <SwitchCamera />
              </button>
            ) : null}
            <button type="button" className="icon-btn" aria-label="Effects and settings" aria-expanded={panelOpen && tab !== 'poses'} onClick={() => openTab('effects')}>
              <SlidersHorizontal />
            </button>
          </div>
        </div>

        {isOn ? (
          <div className="sr-only" aria-live="polite">
            {announce}
          </div>
        ) : null}
      </div>

      <aside className="panel" data-open={panelOpen} aria-label="Studio panel">
        <div className="panel__tabs" role="tablist" aria-label="Studio panel sections">
          {(['poses', 'effects', 'sound'] as const).map((t) => (
            <button key={t} type="button" role="tab" id={`tab-${t}`} aria-controls="panel-body" aria-selected={tab === t} className="panel__tab" onClick={() => setTab(t)}>
              {t === 'poses' ? 'Poses' : t === 'effects' ? 'Effects' : 'Sound'}
            </button>
          ))}
          <button type="button" className="icon-btn panel__close" aria-label="Close panel" onClick={() => setPanelOpen(false)}>
            <X />
          </button>
        </div>
        <div className="panel__body" id="panel-body" role="tabpanel" aria-labelledby={`tab-${tab}`}>
          {tab === 'poses' ? <PoseGuide display={isOn ? display : { kind: 'none' }} /> : null}
          {tab === 'effects' ? (
            <EffectsPanel devices={status?.devices ?? []} deviceId={status?.deviceId ?? null} onSelectCamera={(id) => void engine?.selectCamera(id)} />
          ) : null}
          {tab === 'sound' ? <SoundPanel /> : null}
        </div>
      </aside>

      {photo ? <CaptureDialog photo={photo} onClose={closePhoto} /> : null}
    </div>
  );
}

function StageOverlay({ status, onStart }: { status: StudioStatus | null; onStart: () => void }) {
  const camera = status?.camera ?? 'off';
  if (camera === 'starting') {
    return (
      <div className="stage__overlay">
        <div className="stage__card" role="status">
          <span className="spinner" aria-hidden="true" />
          <h1>Waiting for your camera</h1>
          <p className="prose">If your browser asks for permission, choose Allow.</p>
        </div>
      </div>
    );
  }
  if (camera === 'error' && status?.cameraError) {
    const copy = CAMERA_ERROR_COPY[status.cameraError];
    const retryable = status.cameraError !== 'unsupported' && status.cameraError !== 'insecure';
    return (
      <div className="stage__overlay">
        <div className="stage__card" role="alert">
          <span className="stage__icon stage__icon--error">
            <AlertTriangle aria-hidden="true" />
          </span>
          <h1>{copy.title}</h1>
          <p className="prose">{copy.body}</p>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', justifyContent: 'center' }}>
            {retryable ? (
              <button type="button" className="btn btn--primary" onClick={onStart}>
                <RefreshCw aria-hidden="true" />
                Try again
              </button>
            ) : null}
            <Link to="/help#camera" className="btn btn--ghost">
              Camera help
            </Link>
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="stage__overlay">
      <div className="stage__card">
        <span className="stage__icon">
          <Camera aria-hidden="true" />
        </span>
        <h1>Turn on your camera to start posing</h1>
        <p className="prose">{PRIVACY_STATEMENT}</p>
        <button type="button" className="btn btn--primary btn--lg" onClick={onStart} disabled={!status}>
          <Camera aria-hidden="true" />
          Turn on camera
        </button>
        <p className="prose" style={{ fontSize: '0.875rem' }}>
          Best results: good light, hands fully in view, about an arm’s length away.
        </p>
      </div>
    </div>
  );
}
