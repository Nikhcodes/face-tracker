/**
 * A small kinematic hand model that generates MediaPipe-shaped landmarks.
 *
 * It is used for two things:
 *  1. Unit tests — the classifier is exercised with hands rotated, mirrored,
 *     scaled and jittered, which a fixed set of recorded frames cannot cover.
 *  2. Illustrations — pose glyphs in the UI are rendered from the same specs
 *     the classifier is tested against, so the drawings match the logic.
 *
 * It is a geometric approximation, not a substitute for testing with a real
 * webcam (see README → manual test checklist).
 */
import type { HandObservation, Handedness, Point3 } from './types';
import { add, centroid, cross, dot, normalize, scale, sub } from './geometry';

export interface FingerPose {
  /** Flexion at the knuckle (MCP), degrees. */
  mcp: number;
  /** Flexion at the middle joint (PIP), degrees. */
  pip: number;
  /** Flexion at the end joint (DIP), degrees. */
  dip: number;
  /** Sideways spread, degrees (positive = towards the thumb side). */
  spread?: number;
}

export interface ThumbPose {
  /** Swing across the palm at the base, degrees. */
  cmc: number;
  mcp: number;
  ip: number;
  /** Swing away from the fingers in the palm plane, degrees. */
  abduct?: number;
}

export interface HandSpec {
  index: FingerPose;
  middle: FingerPose;
  ring: FingerPose;
  pinky: FingerPose;
  thumb: ThumbPose;
}

export const F_EXT: FingerPose = { mcp: 4, pip: 4, dip: 2 };
export const F_FOLD: FingerPose = { mcp: 80, pip: 100, dip: 55 };
export const F_HALF: FingerPose = { mcp: 35, pip: 45, dip: 25 };
export const T_EXT: ThumbPose = { cmc: 0, mcp: 6, ip: 6 };
export const T_TUCK: ThumbPose = { cmc: 45, mcp: 45, ip: 35 };
export const T_REST: ThumbPose = { cmc: 30, mcp: 25, ip: 20 };

const spread = (f: FingerPose, s: number): FingerPose => ({ ...f, spread: s });

/** Canonical specs for each live pose (also used to draw glyphs). */
export const SPECS = {
  open: { index: spread(F_EXT, 8), middle: F_EXT, ring: spread(F_EXT, -6), pinky: spread(F_EXT, -14), thumb: T_EXT },
  fist: { index: F_FOLD, middle: F_FOLD, ring: F_FOLD, pinky: F_FOLD, thumb: T_TUCK },
  point: { index: F_EXT, middle: F_FOLD, ring: F_FOLD, pinky: F_FOLD, thumb: T_TUCK },
  peace: { index: spread(F_EXT, 10), middle: spread(F_EXT, -6), ring: F_FOLD, pinky: F_FOLD, thumb: T_TUCK },
  three: { index: spread(F_EXT, 9), middle: F_EXT, ring: spread(F_EXT, -8), pinky: F_FOLD, thumb: T_TUCK },
  rockon: { index: spread(F_EXT, 6), middle: F_FOLD, ring: F_FOLD, pinky: spread(F_EXT, -12), thumb: T_TUCK },
  shaka: { index: F_FOLD, middle: F_FOLD, ring: F_FOLD, pinky: spread(F_EXT, -18), thumb: T_EXT },
  gun: { index: F_EXT, middle: F_FOLD, ring: F_FOLD, pinky: F_FOLD, thumb: { ...T_EXT, abduct: 28 } },
  thumb: { index: F_FOLD, middle: F_FOLD, ring: F_FOLD, pinky: F_FOLD, thumb: T_EXT },
  heartHalf: {
    index: { mcp: 30, pip: 60, dip: 30 },
    middle: { mcp: 30, pip: 60, dip: 30 },
    ring: F_FOLD,
    pinky: F_FOLD,
    thumb: { cmc: 10, mcp: 20, ip: 10 },
  },
} satisfies Record<string, HandSpec>;

/** A hand drawn in a pose illustration. */
export type GlyphHand = { spec: HandSpec; roll?: number; yaw?: number; mirror?: boolean; dx?: number; dy?: number };

export interface Placement {
  /** Hand centre in normalised image coordinates. */
  cx?: number;
  cy?: number;
  /** Image-height units per metre (≈2.2 means a hand fills ~40% of the frame height). */
  size?: number;
  /** Rotation in degrees. roll: in-plane (positive = clockwise on screen); yaw: around vertical; pitch: around horizontal. */
  roll?: number;
  yaw?: number;
  pitch?: number;
  handedness?: Exclude<Handedness, 'Unknown'>;
  /** Frame aspect ratio (width / height). */
  aspect?: number;
  /** Rotations applied *after* roll: turn around the vertical axis, tilt towards/away from the camera. */
  turn?: number;
  tilt?: number;
  /** Uniform landmark jitter in metres (simulates detector noise). */
  noise?: number;
  /** Deterministic seed for the noise. */
  seed?: number;
}

// Canonical frame: right hand, palm facing the camera, fingers pointing up,
// x right, y down, z away from the camera. The thumb is on the +x side.
const PALM_NORMAL: Point3 = { x: 0, y: 0, z: -1 }; // towards the camera
const BASES: Record<'index' | 'middle' | 'ring' | 'pinky', Point3> = {
  index: { x: 0.026, y: -0.083, z: 0 },
  middle: { x: 0.006, y: -0.087, z: 0 },
  ring: { x: -0.012, y: -0.082, z: 0 },
  pinky: { x: -0.028, y: -0.073, z: 0 },
};
const BASE_SPLAY: Record<'index' | 'middle' | 'ring' | 'pinky', number> = { index: 4, middle: 0, ring: -4, pinky: -9 };
const LENGTHS: Record<'index' | 'middle' | 'ring' | 'pinky', [number, number, number]> = {
  index: [0.039, 0.023, 0.019],
  middle: [0.044, 0.027, 0.021],
  ring: [0.041, 0.026, 0.02],
  pinky: [0.032, 0.019, 0.018],
};
const THUMB_CMC: Point3 = { x: 0.021, y: -0.022, z: -0.006 };
const THUMB_LENGTHS: [number, number, number] = [0.038, 0.031, 0.027];
const THUMB_UP: Point3 = normalize({ x: 0.72, y: -0.62, z: -0.3 });
const THUMB_CURL: Point3 = (() => {
  const raw = normalize({ x: -0.75, y: -0.25, z: -0.62 });
  // orthogonalise against THUMB_UP so the swing stays in one plane
  return normalize(sub(raw, scale(THUMB_UP, dot(raw, THUMB_UP))));
})();

const rad = (d: number) => (d * Math.PI) / 180;
const dirInPlane = (u: Point3, n: Point3, deg: number): Point3 => add(scale(u, Math.cos(rad(deg))), scale(n, Math.sin(rad(deg))));

function rotateZ(p: Point3, deg: number): Point3 {
  const c = Math.cos(rad(deg));
  const s = Math.sin(rad(deg));
  return { x: p.x * c - p.y * s, y: p.x * s + p.y * c, z: p.z };
}
function rotateY(p: Point3, deg: number): Point3 {
  const c = Math.cos(rad(deg));
  const s = Math.sin(rad(deg));
  return { x: p.x * c + p.z * s, y: p.y, z: -p.x * s + p.z * c };
}
function rotateX(p: Point3, deg: number): Point3 {
  const c = Math.cos(rad(deg));
  const s = Math.sin(rad(deg));
  return { x: p.x, y: p.y * c - p.z * s, z: p.y * s + p.z * c };
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Builds the 21 canonical-frame points for a spec (right hand, palm to camera). */
export function canonicalPoints(spec: HandSpec): Point3[] {
  const pts: Point3[] = new Array(21);
  pts[0] = { x: 0, y: 0, z: 0 };

  // Thumb
  pts[1] = THUMB_CMC;
  const abduct = spec.thumb.abduct ?? 0;
  const thumbUp = rotateZ(THUMB_UP, abduct);
  const thumbCurl = rotateZ(THUMB_CURL, abduct);
  let angle = spec.thumb.cmc;
  let cur = THUMB_CMC;
  const thumbAngles = [spec.thumb.cmc, spec.thumb.mcp, spec.thumb.ip];
  for (let i = 0; i < 3; i++) {
    if (i > 0) angle += thumbAngles[i];
    cur = add(cur, scale(normalize(dirInPlane(thumbUp, thumbCurl, angle)), THUMB_LENGTHS[i]));
    pts[2 + i] = cur;
  }

  // Fingers
  const order = ['index', 'middle', 'ring', 'pinky'] as const;
  order.forEach((name, fi) => {
    const f = spec[name];
    const base = BASES[name];
    const up = rotateZ({ x: 0, y: -1, z: 0 }, -(BASE_SPLAY[name] + (f.spread ?? 0)));
    // flexion plane: spanned by the finger's up direction and the palm normal
    const n = normalize(cross(cross(up, PALM_NORMAL), up));
    const flexNormal = dot(n, PALM_NORMAL) < 0 ? scale(n, -1) : n;
    const startIdx = 5 + fi * 4;
    pts[startIdx] = base;
    let a = 0;
    let p = base;
    const angles = [f.mcp, f.pip, f.dip];
    for (let j = 0; j < 3; j++) {
      a += angles[j];
      p = add(p, scale(normalize(dirInPlane(up, flexNormal, a)), LENGTHS[name][j]));
      pts[startIdx + 1 + j] = p;
    }
  });
  return pts;
}

/** Generates a HandObservation as MediaPipe would report it for the raw (un-mirrored) frame. */
export function synthHand(spec: HandSpec, placement: Placement = {}): HandObservation {
  const {
    cx = 0.5,
    cy = 0.5,
    size = 2.2,
    roll = 0,
    yaw = 0,
    pitch = 0,
    handedness = 'Right',
    aspect = 16 / 9,
    noise = 0,
    seed = 1,
    turn = 0,
    tilt = 0,
  } = placement;
  const rand = mulberry32(seed);
  let pts = canonicalPoints(spec);
  if (handedness === 'Left') pts = pts.map((p) => ({ x: -p.x, y: p.y, z: p.z }));
  pts = pts.map((p) => rotateX(rotateY(rotateZ(rotateX(rotateY(p, yaw), pitch), roll), turn), tilt));
  if (noise > 0) {
    pts = pts.map((p) => ({
      x: p.x + (rand() * 2 - 1) * noise,
      y: p.y + (rand() * 2 - 1) * noise,
      z: p.z + (rand() * 2 - 1) * noise,
    }));
  }
  const c = centroid(pts);
  const world = pts.map((p) => sub(p, c));
  const image = world.map((p) => ({
    x: (cx * aspect + p.x * size) / aspect,
    y: cy + p.y * size,
    z: (p.z - world[0].z) * size,
  }));
  // MediaPipe's handedness label assumes a mirrored (selfie) image, so for a
  // raw camera frame it reports the opposite hand. Mimic that.
  const reported: Handedness = handedness === 'Right' ? 'Left' : 'Right';
  return { image, world, handedness: reported, score: 0.95 };
}

/** Rotation (roll, degrees) that points the thumb of a spec straight up or down on screen. */
export function rollToPointThumb(spec: HandSpec, target: 'up' | 'down', placement: Placement = {}): number {
  const probe = synthHand(spec, { ...placement, roll: 0, turn: 0, tilt: 0 });
  const v = sub(probe.world[4], probe.world[2]);
  const current = (Math.atan2(v.y, v.x) * 180) / Math.PI;
  const wanted = target === 'up' ? -90 : 90;
  return wanted - current;
}
