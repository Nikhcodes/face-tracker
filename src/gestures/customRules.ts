/**
 * Rules for poses created in the pose editor.
 *
 * A custom pose is a finger pattern (each digit up, down or "doesn't matter")
 * plus optional thumb direction and frame zone. It is matched against the
 * same orientation-independent features the built-in classifier uses.
 */
import type { DigitFeature, HandFeatures } from './features';

export type DigitReq = 'up' | 'down' | 'any';
export const DIGITS = ['thumb', 'index', 'middle', 'ring', 'pinky'] as const;
export type DigitName = (typeof DIGITS)[number];

export interface CustomRule {
  hands: 1 | 2;
  thumb: DigitReq;
  index: DigitReq;
  middle: DigitReq;
  ring: DigitReq;
  pinky: DigitReq;
  /** Only meaningful when the thumb is up. */
  thumbDirection: 'any' | 'up' | 'down';
  /** Where the hand's centre must be (raw frame, so mirroring doesn't matter). */
  zone: 'any' | 'upper' | 'lower';
}

export const DEFAULT_RULE: CustomRule = {
  hands: 1,
  thumb: 'down',
  index: 'up',
  middle: 'up',
  ring: 'up',
  pinky: 'up',
  thumbDirection: 'any',
  zone: 'any',
};

const digit = (f: HandFeatures, name: DigitName): DigitFeature => (name === 'thumb' ? f.thumb : f.fingers[name]);

/** Problems that would make a rule match too much or never match. Empty = valid. */
export function ruleIssues(rule: CustomRule): string[] {
  const issues: string[] = [];
  const set = DIGITS.filter((d) => rule[d] !== 'any').length;
  if (set < 2) issues.push('Set at least two fingers to Up or Down so the pose can be told apart from a relaxed hand.');
  if (rule.thumbDirection !== 'any' && rule.thumb !== 'up') issues.push('Thumb direction only works when the thumb is set to Up.');
  return issues;
}

/** Returns a confidence (0.55–1) if one hand matches the rule's finger pattern, else null. */
export function matchHand(rule: CustomRule, f: HandFeatures): number | null {
  const required: DigitFeature[] = [];
  for (const name of DIGITS) {
    const req = rule[name];
    if (req === 'any') continue;
    const d = digit(f, name);
    if (req === 'up' ? d.state !== 'extended' : d.state !== 'folded') return null;
    required.push(d);
  }
  if (rule.thumbDirection === 'up' && !(f.thumb.vertical < -0.55)) return null;
  if (rule.thumbDirection === 'down' && !(f.thumb.vertical > 0.55)) return null;
  if (rule.zone === 'upper' && !(f.center.y < 0.5)) return null;
  if (rule.zone === 'lower' && !(f.center.y >= 0.5)) return null;
  if (!required.length) return null;
  return 0.55 + 0.45 * Math.min(...required.map((d) => d.certainty));
}

/** Matches a whole frame: one-hand rules match either hand; two-hand rules need both. */
export function matchRule(rule: CustomRule, hands: HandFeatures[]): number | null {
  if (rule.hands === 2) {
    if (hands.length < 2) return null;
    const a = matchHand(rule, hands[0]);
    const b = matchHand(rule, hands[1]);
    return a !== null && b !== null ? Math.min(a, b) : null;
  }
  let best: number | null = null;
  for (const h of hands) {
    const c = matchHand(rule, h);
    if (c !== null && (best === null || c > best)) best = c;
  }
  return best;
}

/** Turns a live hand into a rule ("teach from camera"). Unsure digits become "any". */
export function ruleFromFeatures(f: HandFeatures, hands: 1 | 2 = 1): CustomRule {
  const req = (d: DigitFeature): DigitReq => (d.state === 'extended' ? 'up' : d.state === 'folded' ? 'down' : 'any');
  const thumb = req(f.thumb);
  const thumbDirection = thumb === 'up' && Math.abs(f.thumb.vertical) > 0.7 ? (f.thumb.vertical < 0 ? 'up' : 'down') : 'any';
  return {
    hands,
    thumb,
    index: req(f.fingers.index),
    middle: req(f.fingers.middle),
    ring: req(f.fingers.ring),
    pinky: req(f.fingers.pinky),
    thumbDirection,
    zone: 'any',
  };
}

export function isRule(x: unknown): x is CustomRule {
  if (!x || typeof x !== 'object') return false;
  const r = x as Record<string, unknown>;
  const reqOk = (v: unknown) => v === 'up' || v === 'down' || v === 'any';
  return (
    (r.hands === 1 || r.hands === 2) &&
    DIGITS.every((d) => reqOk(r[d])) &&
    (r.thumbDirection === 'any' || r.thumbDirection === 'up' || r.thumbDirection === 'down') &&
    (r.zone === 'any' || r.zone === 'upper' || r.zone === 'lower')
  );
}
