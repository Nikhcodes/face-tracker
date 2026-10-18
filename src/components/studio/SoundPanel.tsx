import { sound, SoundEngine } from '../../audio/sound';
import { useStore } from '../../lib/useStore';
import { settingsStore, updateSettings } from '../../state/settings';
import { ToggleRow } from './Controls';

/** Turns sound on from a user gesture (required by browsers). */
export async function setSoundEnabled(on: boolean): Promise<void> {
  if (on) {
    const ok = await sound.enable();
    sound.setVolume(settingsStore.get().volume);
    updateSettings({ sound: ok });
    if (ok) sound.play('ui');
    if (ok && settingsStore.get().music) sound.startMusic();
  } else {
    sound.disable();
    updateSettings({ sound: false });
  }
}

export function SoundPanel() {
  const s = useStore(settingsStore);
  if (!SoundEngine.supported()) {
    return <p className="muted">This browser doesn’t support the Web Audio API, so sound is unavailable. Everything else works.</p>;
  }
  return (
    <>
      <section className="panel__section">
        <ToggleRow label="Sound effects" hint="Short synthesized cues for poses and the shutter" checked={s.sound} onChange={(v) => void setSoundEnabled(v)} />
        <ToggleRow
          label="Background music"
          hint={s.sound ? 'A soft generated loop' : 'Turn on sound effects first'}
          checked={s.music && s.sound}
          onChange={(v) => {
            if (v && !s.sound) {
              void setSoundEnabled(true).then(() => {
                updateSettings({ music: true });
                sound.startMusic();
              });
              return;
            }
            updateSettings({ music: v });
            if (v) sound.startMusic();
            else sound.stopMusic();
          }}
        />
      </section>
      <section className="panel__section">
        <label className="field-label" htmlFor="vol">
          Volume
        </label>
        <input
          id="vol"
          className="range"
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={s.volume}
          onChange={(e) => {
            const volume = Number(e.target.value);
            updateSettings({ volume });
            sound.setVolume(volume);
          }}
        />
        <p className="field-hint">Sound is always off when the page loads. All sounds are generated in your browser.</p>
      </section>
    </>
  );
}
