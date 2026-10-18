import { Camera, Check, Hand, PenLine } from 'lucide-react';
import { useEffect, useState } from 'react';
import { HandGlyph } from '../components/HandGlyph';
import { SiteLayout } from '../components/SiteLayout';
import type { PoseDefinition } from '../config/poses';
import { registryStore } from '../config/registry';
import { Link } from '../lib/router';
import { useStore } from '../lib/useStore';
import { unlockedStore } from '../state/progress';

type Filter = 'all' | 'one' | 'two' | 'experimental' | 'custom';

const STATUS_LABEL: Record<PoseDefinition['status'], string> = {
  live: 'Live detection',
  experimental: 'Experimental',
  state: 'Automatic',
  custom: 'Custom',
};

function ReactionPreview({ pose }: { pose: PoseDefinition }) {
  const s = pose.stickers[0];
  if (!s) return <span />;
  if (s.kind === 'stamp') {
    return (
      <span className="pose-card__stamp" aria-hidden="true">
        <HandGlyph hands={pose.glyph} fill="currentColor" outline="#0c0c0d" />
        {s.label}
      </span>
    );
  }
  const src = s.src;
  return (
    <span className="pose-card__sticker">
      {s.kind === 'sprite' ? (
        <span
          role="img"
          aria-label={`${pose.name} sticker`}
          style={{
            display: 'block',
            aspectRatio: `${s.frameWidth} / ${s.frameHeight}`,
            backgroundImage: `url(${src})`,
            backgroundSize: `${s.columns * 100}% auto`,
            backgroundPosition: '0 0',
            borderRadius: 7,
          }}
        />
      ) : (
        <img src={src} alt={`${pose.name} sticker`} width={s.width} height={s.height} />
      )}
    </span>
  );
}

export function Poses() {
  const unlocked = useStore(unlockedStore);
  const { poses } = useStore(registryStore);
  const hasCustom = poses.some((p) => p.status === 'custom');
  const [filter, setFilter] = useState<Filter>('all');
  useEffect(() => {
    document.title = 'Poses — POSE CAM';
  }, []);
  const shown = poses.filter((p) =>
    filter === 'all'
      ? true
      : filter === 'one'
        ? p.hands === 1
        : filter === 'two'
          ? p.hands === 2
          : filter === 'custom'
            ? p.status === 'custom'
            : p.status === 'experimental',
  );

  return (
    <SiteLayout>
      <header className="page-head">
        <h1>Every pose, and what it does</h1>
        <p>
          Live poses are recognized reliably in good light. Experimental ones work, but depend more on angle and framing. Selecting
          a card only shows instructions; detection happens in the camera.
        </p>
        <div className="hero__actions" style={{ marginTop: 0 }}>
          <Link to="/studio" state={{ autostart: true }} className="btn btn--primary">
            <Camera aria-hidden="true" />
            Try them in the camera
          </Link>
          <Link to="/admin" className="btn btn--ghost">
            <PenLine aria-hidden="true" />
            Make your own pose
          </Link>
        </div>
      </header>

      <div className="filter-row" role="group" aria-label="Filter poses">
        {(
          [
            ['all', 'All'],
            ['one', 'One hand'],
            ['two', 'Two hands'],
            ['experimental', 'Experimental'],
            ...(hasCustom ? ([['custom', 'Custom']] as const) : []),
          ] as const
        ).map(([f, label]) => (
          <button key={f} type="button" className="chip-btn" aria-pressed={filter === f} onClick={() => setFilter(f)}>
            {label}
          </button>
        ))}
      </div>

      <ul className="pose-grid">
        {shown.map((p) => (
          <li key={p.id} className="pose-card" id={p.id}>
            <div className="pose-card__visual">
              <span className="pose-card__glyph">
                {p.glyph.length ? (
                  <HandGlyph hands={p.glyph} fill="var(--text)" outline="#1e1e1a" title={`Hand shape for ${p.name}`} />
                ) : (
                  <Hand size={56} aria-hidden="true" style={{ margin: '0 auto', opacity: 0.35 }} />
                )}
              </span>
              <ReactionPreview pose={p} />
              {p.captions[0] ? (
                <span className="tape" aria-label={`Caption: ${p.captions[0]}`}>
                  {p.captions[0]}
                </span>
              ) : null}
            </div>
            <div className="pose-card__body">
              <div className="pose-card__title">
                <h2>{p.name}</h2>
                <span
                  className={`badge ${p.status === 'live' ? 'badge--live' : p.status === 'experimental' ? 'badge--exp' : p.status === 'custom' ? 'badge--custom' : ''}`}
                >
                  {STATUS_LABEL[p.status]}
                </span>
              </div>
              <p className="pose-card__blurb">{p.blurb}</p>
              <p className="pose-card__how">
                <Hand aria-hidden="true" />
                <span>{p.howTo}</span>
              </p>
              {unlocked.has(p.id) ? (
                <span className="pose-card__found">
                  <Check aria-hidden="true" />
                  You’ve done this one
                </span>
              ) : null}
            </div>
          </li>
        ))}
      </ul>
    </SiteLayout>
  );
}
