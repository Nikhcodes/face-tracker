/**
 * Orientation-independent hand features.
 *
 * Finger states use joint angles and distance ratios measured on MediaPipe's
 * metric *world* landmarks, so they do not depend on whether the hand is
 * upright, sideways, mirrored or tilted. Image landmarks (corrected for the
 * frame's aspect ratio) are only used for *where* the hand is.
 */
import { angleBetween, centroid, dist, dot, normalize, ramp, scale, sub, toIsotropic } from './geometry';
import { FINGER_JOINTS, FINGERS, LM, type FingerName, type HandObservation, type Point3 } from './types';

export type DigitState = 'extended' | 'folded' | 'uncertain';

export interface DigitFeature {
  state: DigitState;
  /** 0 = clearly folded, 1 = clearly extended. */
  extension: number;
  /** Total bend along the chain, degrees. */
  bend: number;
  /** How far the score sits from the decision boundary, 0..1. */
  certainty: number;
}

export interface ThumbFeature extends DigitFeature {
  /** Unit direction MCP → tip in world space (y down). */
  direction: Point3;
  /** Vertical component of direction: -1 = straight up, +1 = straight down. */
  vertical: number;
  /** True when the thumb tip is the highest (or lowest) point of the hand. */
  isTopmost: boolean;
  isBottommost: boolean;
}

export interface HandFeatures {
  fingers: Record<FingerName, DigitFeature>;
  thumb: ThumbFeature;
  /** Wrist → middle knuckle, metres. */
  palmSize: number;
  /** Mean of all image landmarks (normalised coordinates). */
  center: { x: number; y: number };
  /** Palm size in isotropic image units (fraction of frame height). */
  imagePalmSize: number;
  /** Unit direction of the index finger (MCP → tip) in world space. */
  indexDirection: Point3;
}

export const THRESHOLDS = {
  extended: 0.66,
  folded: 0.36,
} as const;

function classifyDigit(extension: number, bend: number): DigitFeature {
  let state: DigitState = 'uncertain';
  if (extension >= THRESHOLDS.extended) state = 'extended';
  else if (extension <= THRESHOLDS.folded) state = 'folded';
  const mid = (THRESHOLDS.extended + THRESHOLDS.folded) / 2;
  const half = (THRESHOLDS.extended - THRESHOLDS.folded) / 2;
  const certainty = Math.min(1, Math.abs(extension - mid) / (half + 0.25));
  return { state, extension, bend, certainty };
}

/** Removes the component of v along the (unit) axis. */
const reject = (v: Point3, axis: Point3) => sub(v, scale(axis, dot(v, axis)));

export function fingerFeature(world: readonly Point3[], finger: FingerName): DigitFeature {
  const [mcp, pip, dip, tip] = FINGER_JOINTS[finger].map((i) => world[i]);
  const wrist = world[LM.WRIST];
  // Knuckle flexion is measured with sideways spread removed, so a splayed
  // pinky does not look half-bent.
  const lateral = normalize(sub(world[LM.INDEX_MCP], world[LM.PINKY_MCP]));
  const bend =
    angleBetween(reject(sub(mcp, wrist), lateral), reject(sub(pip, mcp), lateral)) +
    angleBetween(sub(pip, mcp), sub(dip, pip)) +
    angleBetween(sub(dip, pip), sub(tip, dip));
  // A straight finger puts its tip ~1.9× further from the wrist than its knuckle.
  const reach = dist(tip, wrist) / Math.max(dist(mcp, wrist), 1e-6);
  const straightness = 1 - ramp(bend, 45, 130);
  const reachScore = ramp(reach, 1.15, 1.7);
  return classifyDigit(0.6 * straightness + 0.4 * reachScore, bend);
}

export function thumbFeature(world: readonly Point3[], palmSize: number): ThumbFeature {
  const cmc = world[LM.THUMB_CMC];
  const mcp = world[LM.THUMB_MCP];
  const ip = world[LM.THUMB_IP];
  const tip = world[LM.THUMB_TIP];
  const bend = angleBetween(sub(mcp, cmc), sub(ip, mcp)) + angleBetween(sub(ip, mcp), sub(tip, ip));
  // Tucked thumbs rest on the fingers / palm.
  const tuck =
    Math.min(
      dist(tip, world[LM.INDEX_PIP]),
      dist(tip, world[LM.MIDDLE_PIP]),
      dist(tip, world[LM.RING_PIP]),
      dist(tip, world[LM.MIDDLE_MCP]),
      dist(tip, world[LM.RING_MCP]),
      dist(tip, world[LM.INDEX_DIP]),
      dist(tip, world[LM.MIDDLE_DIP]),
    ) / palmSize;
  // Extended thumbs reach away from the pinky side of the hand.
  const away = dist(tip, world[LM.PINKY_MCP]) / Math.max(dist(mcp, world[LM.PINKY_MCP]), 1e-6);
  const extension =
    0.35 * (1 - ramp(bend, 25, 75)) + 0.35 * ramp(tuck, 0.3, 0.62) + 0.3 * ramp(away, 1.05, 1.4);
  const base = classifyDigit(extension, bend);
  const direction = normalize(sub(tip, mcp));
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < world.length; i++) {
    if (i === LM.THUMB_TIP) continue;
    minY = Math.min(minY, world[i].y);
    maxY = Math.max(maxY, world[i].y);
  }
  return {
    ...base,
    direction,
    vertical: direction.y,
    isTopmost: tip.y < minY,
    isBottommost: tip.y > maxY,
  };
}

export function computeFeatures(hand: HandObservation, aspect: number): HandFeatures {
  const w = hand.world;
  const palmSize = Math.max(dist(w[LM.WRIST], w[LM.MIDDLE_MCP]), 1e-6);
  const fingers = {} as Record<FingerName, DigitFeature>;
  for (const f of FINGERS) fingers[f] = fingerFeature(w, f);
  const c = centroid(hand.image);
  const isoWrist = toIsotropic(hand.image[LM.WRIST], aspect);
  const isoMid = toIsotropic(hand.image[LM.MIDDLE_MCP], aspect);
  return {
    fingers,
    thumb: thumbFeature(w, palmSize),
    palmSize,
    center: { x: c.x, y: c.y },
    imagePalmSize: Math.hypot(isoWrist.x - isoMid.x, isoWrist.y - isoMid.y),
    indexDirection: normalize(sub(w[LM.INDEX_TIP], w[LM.INDEX_MCP])),
  };
}

/** Compact string like "T:e I:e M:f R:f P:f" — handy in the debug overlay. */
export function describeFeatures(f: HandFeatures): string {
  const s = (d: DigitFeature) => (d.state === 'extended' ? 'E' : d.state === 'folded' ? 'f' : '?');
  return `T${s(f.thumb)} I${s(f.fingers.index)} M${s(f.fingers.middle)} R${s(f.fingers.ring)} P${s(f.fingers.pinky)}`;
}
