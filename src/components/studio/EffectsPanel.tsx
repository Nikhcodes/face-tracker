import { PenLine, RotateCcw } from 'lucide-react';
import { Link } from '../../lib/router';
import { FILTER_LABELS } from '../../render/effects';
import { useStore } from '../../lib/useStore';
import { hasAllLivePoses, unlockedStore } from '../../state/progress';
import { FILTERS, resetEffects, settingsStore, updateSettings, type FilterId } from '../../state/settings';
import { pushToast } from '../../state/toasts';
import { Segmented, ToggleRow } from './Controls';

const SWATCH: Record<FilterId, string> = {
  original: 'linear-gradient(135deg,#d9cbb8,#6f5f4f)',
  cinematic: 'linear-gradient(135deg,#f0a35e,#0f5e66)',
  mono: 'linear-gradient(135deg,#eee,#333)',
  warm: 'linear-gradient(135deg,#ffd29a,#b8562a)',
  cool: 'linear-gradient(135deg,#c8e2ff,#2a5db8)',
};

export function EffectsPanel({
  devices,
  deviceId,
  onSelectCamera,
}: {
  devices: { id: string; label: string }[];
  deviceId: string | null;
  onSelectCamera: (id: string) => void;
}) {
  const s = useStore(settingsStore);
  const unlocked = useStore(unlockedStore);
  return (
    <>
      <section className="panel__section" aria-labelledby="fx-look">
        <h3 id="fx-look">Look</h3>
        <div className="looks" role="radiogroup" aria-labelledby="fx-look">
          {FILTERS.map((f) => (
            <button key={f} type="button" role="radio" className="look" aria-checked={s.filter === f} onClick={() => updateSettings({ filter: f })}>
              <span className="look__swatch" style={{ background: SWATCH[f] }} aria-hidden="true" />
              {FILTER_LABELS[f]}
            </button>
          ))}
        </div>
      </section>

      <section className="panel__section" aria-labelledby="fx-layers">
        <h3 id="fx-layers">Layers</h3>
        <ToggleRow label="Stickers" hint="Meme cards for each pose" checked={s.stickers} onChange={(v) => updateSettings({ stickers: v })} />
        <ToggleRow label="Reactions" hint="Captions, confetti and other effects" checked={s.reactions} onChange={(v) => updateSettings({ reactions: v })} />
        <ToggleRow label="Vignette" checked={s.vignette} onChange={(v) => updateSettings({ vignette: v })} />
        <ToggleRow label="Film grain" checked={s.grain} onChange={(v) => updateSettings({ grain: v })} />
        {hasAllLivePoses(unlocked) ? (
          <ToggleRow label="Gold frame" hint="Unlocked for finding every live pose" checked={s.goldFrame} onChange={(v) => updateSettings({ goldFrame: v })} />
        ) : null}
        {s.chaos ? <ToggleRow label="Chaos mode" hint="You found the secret. Extra absurd." checked={s.chaos} onChange={(v) => updateSettings({ chaos: v })} /> : null}
      </section>

      <section className="panel__section" aria-labelledby="fx-camera">
        <h3 id="fx-camera">Camera</h3>
        <ToggleRow label="Mirror preview" hint="Like a mirror. Photos match what you see." checked={s.mirror} onChange={(v) => updateSettings({ mirror: v })} />
        <ToggleRow
          label="Three-finger photo timer"
          hint="Hold up three fingers to start a 3-second countdown"
          checked={s.photoTimer}
          onChange={(v) => updateSettings({ photoTimer: v })}
        />
        {devices.length > 1 ? (
          <div style={{ marginTop: 12 }}>
            <label className="field-label" htmlFor="cam-select">
              Camera
            </label>
            <select id="cam-select" className="select" value={deviceId ?? ''} onChange={(e) => onSelectCamera(e.target.value)}>
              {devices.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
        ) : null}
        <div style={{ marginTop: 16 }}>
          <Segmented
            label="Performance"
            value={s.quality}
            onChange={(quality) => updateSettings({ quality })}
            options={[
              { value: 'performance', label: 'Smooth' },
              { value: 'balanced', label: 'Balanced' },
              { value: 'high', label: 'Sharp' },
            ]}
          />
          <p className="field-hint">Smooth uses a smaller image and fewer detections. Sharp makes bigger photos but needs a faster device.</p>
        </div>
        <div style={{ marginTop: 16 }}>
          <Segmented
            label="Photo format"
            value={s.format}
            onChange={(format) => updateSettings({ format })}
            options={[
              { value: 'png', label: 'PNG' },
              { value: 'jpeg', label: 'JPEG' },
            ]}
          />
        </div>
      </section>

      <section className="panel__section" aria-labelledby="fx-access">
        <h3 id="fx-access">Motion and testing</h3>
        <Segmented
          label="Animation"
          value={s.motion}
          onChange={(motion) => updateSettings({ motion })}
          options={[
            { value: 'system', label: 'System' },
            { value: 'reduced', label: 'Reduced' },
            { value: 'full', label: 'Full' },
          ]}
        />
        <div style={{ marginTop: 8 }}>
          <ToggleRow label="Show hand landmarks" hint="Draws the 21 tracked points (also in photos)" checked={s.landmarks} onChange={(v) => updateSettings({ landmarks: v })} />
          <ToggleRow label="Debug info" hint="Finger states, confidence and timing" checked={s.debug} onChange={(v) => updateSettings({ debug: v })} />
        </div>
      </section>

      <section className="panel__section">
        <button
          type="button"
          className="btn btn--ghost btn--sm"
          onClick={() => {
            settingsStore.set(resetEffects(settingsStore.get()));
            pushToast({ title: 'Effects reset' });
          }}
        >
          <RotateCcw aria-hidden="true" />
          Reset effects
        </button>
        <Link to="/admin" className="btn btn--ghost btn--sm" style={{ marginLeft: 8 }}>
          <PenLine aria-hidden="true" />
          Edit poses
        </Link>
      </section>
    </>
  );
}
