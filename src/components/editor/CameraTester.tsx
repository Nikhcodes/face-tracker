import { Camera, CameraOff, Check, Copy, Hand } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import { CAMERA_ERROR_COPY } from '../../camera/camera';
import type { PoseDefinition } from '../../config/poses';
import { registryStore } from '../../config/registry';
import { classifyFrame } from '../../gestures/classify';
import { ruleFromFeatures, type CustomRule } from '../../gestures/customRules';
import { computeFeatures, type DigitFeature } from '../../gestures/features';
import { HandProbe, type ProbeState } from '../../editor/HandProbe';

const INITIAL: ProbeState = { status: 'off', error: null, hands: [], aspect: 4 / 3 };

function Digit({ name, d }: { name: string; d: DigitFeature }) {
  const word = d.state === 'extended' ? 'Up' : d.state === 'folded' ? 'Folded' : 'Unsure';
  return (
    <li className="ed-digit" data-state={d.state}>
      <span>{name}</span>
      <strong>{word}</strong>
    </li>
  );
}

export function CameraTester({ pose, onCopyShape }: { pose: PoseDefinition; onCopyShape?: (rule: CustomRule) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const probeRef = useRef<HandProbe | null>(null);
  const [state, setState] = useState<ProbeState>(INITIAL);

  useEffect(() => {
    const probe = new HandProbe(videoRef.current!, canvasRef.current!);
    probe.onUpdate = setState;
    probeRef.current = probe;
    return () => probe.dispose();
  }, []);

  const running = state.status === 'on' || state.status === 'loading-model' || state.status === 'starting';
  const feats = state.hands.slice(0, 2).map((h) => computeFeatures(h, state.aspect));
  const first = feats[0];
  const reg = registryStore.get();
  const result = state.hands.length ? classifyFrame({ hands: state.hands, aspect: state.aspect }, { custom: reg.custom, disabled: reg.disabled }) : null;
  const matches = result?.kind === 'pose' && result.pose === pose.id;
  const detectedName = result?.kind === 'pose' ? (reg.byId.get(result.pose)?.name ?? result.pose) : null;
  const disabled = reg.disabled.has(pose.id);

  return (
    <section className="ed-tester" aria-labelledby="tester-title">
      <div className="ed-tester__head">
        <h3 id="tester-title">Test with your camera</h3>
        {running ? (
          <button type="button" className="btn btn--sm btn--ghost" onClick={() => probeRef.current?.stop()}>
            <CameraOff aria-hidden="true" />
            Stop
          </button>
        ) : null}
      </div>
      <div className="ed-tester__view" data-on={running}>
        <video ref={videoRef} className="stage__video" playsInline muted aria-hidden="true" />
        <canvas ref={canvasRef} aria-label="Camera test view with hand landmarks" role="img" />
        {!running ? (
          <div className="ed-tester__idle">
            {state.status === 'error' && state.error ? (
              <p className="ed-help">{state.error === 'model' ? 'Hand tracking couldn’t load. Check your connection and try again.' : CAMERA_ERROR_COPY[state.error].title}</p>
            ) : (
              <p className="ed-help">See what the tracker reads from your hand, and whether this pose is detected.</p>
            )}
            <button type="button" className="btn btn--sm btn--primary" onClick={() => void probeRef.current?.start()}>
              <Camera aria-hidden="true" />
              Turn on camera
            </button>
          </div>
        ) : null}
        {state.status === 'loading-model' ? <span className="ed-tester__badge">Loading hand tracking…</span> : null}
      </div>

      {state.status === 'on' ? (
        <div className="ed-tester__readout" aria-live="polite">
          {first ? (
            <>
              <ul className="ed-digits" aria-label="Finger states for the first hand">
                <Digit name="Thumb" d={first.thumb} />
                <Digit name="Index" d={first.fingers.index} />
                <Digit name="Middle" d={first.fingers.middle} />
                <Digit name="Ring" d={first.fingers.ring} />
                <Digit name="Pinky" d={first.fingers.pinky} />
              </ul>
              <p className="ed-match" data-match={matches}>
                {matches ? <Check aria-hidden="true" /> : <Hand aria-hidden="true" />}
                {disabled
                  ? 'This pose is switched off, so it can’t be detected.'
                  : matches
                    ? `${pose.name} detected`
                    : detectedName
                      ? `Detected ${detectedName} instead`
                      : 'Not matching yet'}
              </p>
              {onCopyShape ? (
                <button
                  type="button"
                  className="btn btn--sm"
                  onClick={() => onCopyShape(ruleFromFeatures(first, feats.length >= 2 ? 2 : 1))}
                >
                  <Copy aria-hidden="true" />
                  Copy my hand shape
                </button>
              ) : null}
            </>
          ) : (
            <p className="ed-help">Show a hand to the camera.</p>
          )}
        </div>
      ) : null}
    </section>
  );
}
