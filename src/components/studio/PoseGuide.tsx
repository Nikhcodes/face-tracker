import { Check, PenLine } from 'lucide-react';
import { Link } from '../../lib/router';
import { livePoseIds, registryStore } from '../../config/registry';
import type { DisplayState } from '../../gestures/stabilizer';
import { useStore } from '../../lib/useStore';
import { unlockedStore } from '../../state/progress';
import { HandGlyph } from '../HandGlyph';

export function PoseGuide({ display }: { display: DisplayState }) {
  const unlocked = useStore(unlockedStore);
  const registry = useStore(registryStore);
  const LIVE_POSES = livePoseIds(registry);
  const found = LIVE_POSES.filter((id) => unlocked.has(id)).length;
  const activeId = display.kind === 'pose' ? display.pose : display.kind === 'idle' ? 'idle' : null;
  return (
    <div>
      <div className="guide__progress">
        <span>
          {found} of {LIVE_POSES.length} live poses found
        </span>
      </div>
      <div className="meter" role="progressbar" aria-label="Live poses found" aria-valuemin={0} aria-valuemax={LIVE_POSES.length} aria-valuenow={found}>
        <span style={{ width: `${(found / LIVE_POSES.length) * 100}%` }} />
      </div>
      <ul className="guide">
        {registry.poses.map((p) => (
          <li key={p.id} className="guide__item" data-active={activeId === p.id}>
            <span className="guide__glyph">
              {p.glyph.length ? <HandGlyph hands={p.glyph} fill="var(--text)" outline="#22221f" /> : <span aria-hidden="true">…</span>}
            </span>
            <span>
              <span className="guide__name">
                {p.name}
                {p.status === 'experimental' ? <span className="badge badge--exp">Experimental</span> : null}
                {p.status === 'custom' ? <span className="badge badge--custom">Custom</span> : null}
                {unlocked.has(p.id) ? <Check className="guide__found" aria-label="Found" /> : null}
                {activeId === p.id ? <span className="sr-only">(detected now)</span> : null}
              </span>
              <span className="guide__how">{p.howTo}</span>
            </span>
          </li>
        ))}
      </ul>
      <Link to="/admin" className="btn btn--ghost btn--sm" style={{ marginTop: 16 }}>
        <PenLine aria-hidden="true" />
        Make your own pose
      </Link>
    </div>
  );
}
