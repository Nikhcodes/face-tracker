import { Ban, CloudRain, Heart, Music, PartyPopper, Play, Sparkles, Star } from 'lucide-react';
import type { ComponentType, SVGProps } from 'react';
import { sound } from '../../audio/sound';
import { PARTICLES, SOUNDS, type ParticleChoice, type SoundCue } from '../../config/pack';
import { settingsStore } from '../../state/settings';
import { ToggleRow } from '../studio/Controls';

const PARTICLE_META: Record<ParticleChoice, { label: string; Icon: ComponentType<SVGProps<SVGSVGElement>> }> = {
  none: { label: 'None', Icon: Ban },
  sparkles: { label: 'Sparkles', Icon: Sparkles },
  stars: { label: 'Stars', Icon: Star },
  confetti: { label: 'Confetti', Icon: PartyPopper },
  hearts: { label: 'Hearts', Icon: Heart },
  notes: { label: 'Notes', Icon: Music },
  rain: { label: 'Rain', Icon: CloudRain },
};

const SOUND_LABEL: Record<SoundCue, string> = {
  pop: 'Pop',
  chime: 'Chime',
  boom: 'Boom',
  pew: 'Pew',
  sad: 'Sad trombone',
  sparkle: 'Sparkle',
  none: 'Silent',
};

export function ReactionEditor({
  particles,
  shake,
  spotlight,
  cue,
  onChange,
}: {
  particles: ParticleChoice;
  shake: boolean;
  spotlight: boolean;
  cue: SoundCue;
  onChange: (p: { particles?: ParticleChoice; shake?: boolean; spotlight?: boolean; sound?: SoundCue }) => void;
}) {
  const preview = async (c: SoundCue) => {
    if (c === 'none') return;
    // previews play even when app sound is off, at the chosen volume — only on click
    const wasOn = sound.enabled;
    await sound.enable();
    sound.setVolume(settingsStore.get().volume);
    sound.play(c);
    if (!wasOn) setTimeout(() => sound.disable(), 900);
  };
  return (
    <div className="ed-reaction">
      <div>
        <span className="field-label" id="fx-particles">
          Effect
        </span>
        <div className="ed-tiles" role="radiogroup" aria-labelledby="fx-particles">
          {PARTICLES.map((p) => {
            const { label, Icon } = PARTICLE_META[p];
            return (
              <button key={p} type="button" role="radio" aria-checked={particles === p} className="ed-tile" onClick={() => onChange({ particles: p })}>
                <Icon aria-hidden="true" />
                {label}
              </button>
            );
          })}
        </div>
      </div>
      <div>
        <ToggleRow label="Screen shake" hint="A short jolt when the pose lands (off with reduced motion)" checked={shake} onChange={(v) => onChange({ shake: v })} />
        <ToggleRow label="Spotlight" hint="Darkens the edges while the pose is held" checked={spotlight} onChange={(v) => onChange({ spotlight: v })} />
      </div>
      <div>
        <span className="field-label" id="fx-sound">
          Sound
        </span>
        <div className="ed-sounds" role="radiogroup" aria-labelledby="fx-sound">
          {SOUNDS.map((s) => (
            <button
              key={s}
              type="button"
              role="radio"
              aria-checked={cue === s}
              className="ed-sound"
              onClick={() => {
                onChange({ sound: s });
                void preview(s);
              }}
            >
              {s !== 'none' ? <Play aria-hidden="true" /> : null}
              {SOUND_LABEL[s]}
            </button>
          ))}
        </div>
        <p className="ed-help">Choosing a sound plays a preview. In the camera, sounds only play when sound is turned on.</p>
      </div>
    </div>
  );
}
