/**
 * Frame-level gesture classification.
 *
 * Pure functions: landmarks in, a single verdict out. Temporal smoothing
 * happens in stabilizer.ts.
 */
import { computeFeatures, type DigitFeature, type HandFeatures } from './features';
import { angleBetween, dist, toIsotropic } from './geometry';
import { matchRule, type CustomRule } from './customRules';
import type { CustomPoseId, DetectablePoseId, GesturePoseId } from './poseIds';
import { LM, type FrameObservation, type HandObservation } from './types';

/** Per-hand shape. `fist` is not a pose on its own; it feeds `think` and `rodrick`. */
export type HandShape =
  | 'open'
  | 'peace'
  | 'three'
  | 'rockon'
  | 'shaka'
  | 'gun'
  | 'point'
  | 'thumbsup'
  | 'thumbsdown'
  | 'fist';

export interface HandVerdict {
  shape: HandShape | null;
  confidence: number;
  /** True when some digit sat between "extended" and "folded". */
  uncertain: boolean;
  features: HandFeatures;
}

export type UnknownReason = 'unrecognized' | 'uncertain' | 'ambiguous';

export type FrameResult =
  | { kind: 'none' }
  | { kind: 'unknown'; reason: UnknownReason; hands: HandVerdict[] }
  | { kind: 'pose'; pose: DetectablePoseId; confidence: number; hands: HandVerdict[] };

export interface ClassifyOptions {
  /** Editor-made poses. They are checked before built-in poses. */
  custom?: ReadonlyArray<{ id: CustomPoseId; rule: CustomRule }>;
  /** Built-in poses switched off in the editor. */
  disabled?: ReadonlySet<string>;
}

export const MIN_HAND_SCORE = 0.5;

const ext = (d: DigitFeature) => d.state === 'extended';
const fold = (d: DigitFeature) => d.state === 'folded';
const conf = (...digits: DigitFeature[]) => 0.55 + 0.45 * Math.min(...digits.map((d) => d.certainty));

export function classifyHand(hand: HandObservation, aspect: number): HandVerdict {
  const f = computeFeatures(hand, aspect);
  const { index: I, middle: M, ring: R, pinky: P } = f.fingers;
  const T = f.thumb;
  const uncertain = [I, M, R, P, T].some((d) => d.state === 'uncertain');
  const v = (shape: HandShape, ...digits: DigitFeature[]): HandVerdict => ({
    shape,
    confidence: conf(...digits),
    uncertain,
    features: f,
  });

  if (ext(I) && ext(M) && ext(R) && ext(P) && !fold(T)) return v('open', I, M, R, P);
  if (ext(I) && ext(M) && ext(R) && fold(P)) return v('three', I, M, R, P);
  if (ext(I) && ext(M) && fold(R) && fold(P)) return v('peace', I, M, R, P);
  if (ext(I) && fold(M) && fold(R) && ext(P)) return v('rockon', I, M, R, P);
  if (fold(I) && fold(M) && fold(R) && ext(P) && ext(T)) return v('shaka', I, M, R, P, T);
  if (ext(I) && fold(M) && fold(R) && fold(P)) {
    const thumbAngle = angleBetween(T.direction, f.indexDirection);
    if (ext(T) && thumbAngle > 45) return v('gun', I, M, R, P, T);
    return v('point', I, M, R, P);
  }
  if (fold(I) && fold(M) && fold(R) && fold(P)) {
    if (ext(T) && T.vertical < -0.6 && T.isTopmost) return v('thumbsup', I, M, R, P, T);
    if (ext(T) && T.vertical > 0.6 && T.isBottommost) return v('thumbsdown', I, M, R, P, T);
    if (!ext(T)) return v('fist', I, M, R, P);
  }
  return { shape: null, confidence: 0, uncertain, features: f };
}

const SINGLE_SHAPE_TO_POSE: Partial<Record<HandShape, GesturePoseId>> = {
  open: 'hello',
  peace: 'peace',
  three: 'three',
  rockon: 'rockon',
  shaka: 'shaka',
  gun: 'fingergun',
  point: 'point',
  thumbsup: 'thumbsup',
  thumbsdown: 'thumbsdown',
};

/** Fist held in the lower-middle of the frame, where a chin usually is. */
export function isThinkPosition(f: HandFeatures): boolean {
  return f.center.y > 0.45 && f.center.y < 0.92 && f.center.x > 0.22 && f.center.x < 0.78;
}

/** Two hands forming a heart: index tips touching, thumb tips touching, index above thumbs. */
export function isHeart(a: HandObservation, b: HandObservation, aspect: number, fa: HandFeatures, fb: HandFeatures): boolean {
  const iso = (h: HandObservation, i: number) => toIsotropic(h.image[i], aspect);
  const scale = (fa.imagePalmSize + fb.imagePalmSize) / 2;
  const indexGap = dist(iso(a, LM.INDEX_TIP), iso(b, LM.INDEX_TIP)) / scale;
  const thumbGap = dist(iso(a, LM.THUMB_TIP), iso(b, LM.THUMB_TIP)) / scale;
  const wristGap = dist(iso(a, LM.WRIST), iso(b, LM.WRIST)) / scale;
  const indexY = (a.image[LM.INDEX_TIP].y + b.image[LM.INDEX_TIP].y) / 2;
  const thumbY = (a.image[LM.THUMB_TIP].y + b.image[LM.THUMB_TIP].y) / 2;
  const lift = (thumbY - indexY) / scale;
  return indexGap < 0.45 && thumbGap < 0.45 && wristGap > 0.9 && lift > 0.5;
}

export function classifyFrame(frame: FrameObservation, options: ClassifyOptions = {}): FrameResult {
  const hands = frame.hands.filter((h) => h.score >= MIN_HAND_SCORE).slice(0, 2);
  if (hands.length === 0) return { kind: 'none' };
  const verdicts = hands.map((h) => classifyHand(h, frame.aspect));
  const off = options.disabled ?? new Set<string>();
  const on = (pose: GesturePoseId) => !off.has(pose);

  // Editor poses first: two-hand rules (when two hands are visible), then one-hand rules.
  if (options.custom?.length) {
    const feats = verdicts.map((v) => v.features);
    let best: { pose: CustomPoseId; confidence: number; hands: number } | null = null;
    for (const c of options.custom) {
      const conf = matchRule(c.rule, feats);
      if (conf === null) continue;
      if (!best || c.rule.hands > best.hands || (c.rule.hands === best.hands && conf > best.confidence)) {
        best = { pose: c.id, confidence: conf, hands: c.rule.hands };
      }
    }
    if (best) return { kind: 'pose', pose: best.pose, confidence: best.confidence, hands: verdicts };
  }

  if (verdicts.length === 2) {
    const [a, b] = verdicts;
    if (on('heart') && isHeart(hands[0], hands[1], frame.aspect, a.features, b.features)) {
      return { kind: 'pose', pose: 'heart', confidence: 0.75, hands: verdicts };
    }
    const both = (s: HandShape) => a.shape === s && b.shape === s;
    const pair = Math.min(a.confidence, b.confidence);
    if (on('doublethumbs') && both('thumbsup')) return { kind: 'pose', pose: 'doublethumbs', confidence: pair, hands: verdicts };
    if (on('celebrate') && both('open')) return { kind: 'pose', pose: 'celebrate', confidence: pair, hands: verdicts };
    if (on('rodrick') && both('fist') && a.features.center.y < 0.62 && b.features.center.y < 0.62) {
      return { kind: 'pose', pose: 'rodrick', confidence: pair, hands: verdicts };
    }
  }

  const candidates: { pose: GesturePoseId; confidence: number }[] = [];
  for (const v of verdicts) {
    if (!v.shape) continue;
    if (v.shape === 'fist') {
      if (on('think') && verdicts.length === 1 && isThinkPosition(v.features)) {
        candidates.push({ pose: 'think', confidence: v.confidence * 0.9 });
      }
      continue;
    }
    const pose = SINGLE_SHAPE_TO_POSE[v.shape];
    if (pose && on(pose)) candidates.push({ pose, confidence: v.confidence });
  }

  if (candidates.length === 0) {
    const reason: UnknownReason = verdicts.some((v) => v.uncertain && !v.shape) ? 'uncertain' : 'unrecognized';
    return { kind: 'unknown', reason, hands: verdicts };
  }
  if (candidates.length === 2 && candidates[0].pose !== candidates[1].pose) {
    const [c1, c2] = candidates.sort((x, y) => y.confidence - x.confidence);
    if (c1.confidence - c2.confidence < 0.08) return { kind: 'unknown', reason: 'ambiguous', hands: verdicts };
    return { kind: 'pose', pose: c1.pose, confidence: c1.confidence, hands: verdicts };
  }
  const best = candidates.reduce((x, y) => (y.confidence > x.confidence ? y : x));
  return { kind: 'pose', pose: best.pose, confidence: best.confidence, hands: verdicts };
}
