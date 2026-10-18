import { CircleDashed, Hand, HelpCircle, MoonStar, Sparkles, Timer } from 'lucide-react';
import { getPose } from '../../config/registry';
import type { DisplayState } from '../../gestures/stabilizer';

export function describeDisplay(d: DisplayState): { text: string; detail?: string } {
  switch (d.kind) {
    case 'pose': {
      const p = getPose(d.pose);
      return { text: p.label, detail: p.status === 'experimental' ? 'Experimental' : p.status === 'custom' ? 'Custom' : undefined };
    }
    case 'idle':
      return { text: 'Idle, waiting for a pose' };
    case 'none':
      return { text: 'Show a hand to the camera' };
    case 'unknown':
      return d.reason === 'uncertain'
        ? { text: 'Hand found, make the pose clearer' }
        : d.reason === 'ambiguous'
          ? { text: 'Two different poses, try one' }
          : { text: 'Hand found, no pose yet' };
  }
}

export function StatusChip({ display, countdown, modelReady }: { display: DisplayState; countdown: number | null; modelReady: boolean }) {
  if (countdown !== null) {
    return (
      <span className="status-chip" data-kind="countdown">
        <Timer aria-hidden="true" />
        <span className="status-chip__text">Photo in {countdown}…</span>
      </span>
    );
  }
  if (!modelReady) {
    return (
      <span className="status-chip" data-kind="loading">
        <CircleDashed aria-hidden="true" />
        <span className="status-chip__text">Hand tracking starting…</span>
      </span>
    );
  }
  const { text, detail } = describeDisplay(display);
  const Icon = display.kind === 'pose' ? Sparkles : display.kind === 'idle' ? MoonStar : display.kind === 'unknown' ? HelpCircle : Hand;
  return (
    <span className="status-chip" data-kind={display.kind}>
      <Icon aria-hidden="true" />
      <span className="status-chip__text">{text}</span>
      {detail ? <span className="badge">{detail}</span> : null}
    </span>
  );
}
