import type { Point3 } from './types';

export const sub = (a: Point3, b: Point3): Point3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const add = (a: Point3, b: Point3): Point3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const scale = (a: Point3, s: number): Point3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const dot = (a: Point3, b: Point3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const cross = (a: Point3, b: Point3): Point3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const length = (a: Point3): number => Math.hypot(a.x, a.y, a.z);
export const dist = (a: Point3, b: Point3): number => length(sub(a, b));
export const normalize = (a: Point3): Point3 => {
  const l = length(a);
  return l > 1e-9 ? scale(a, 1 / l) : { x: 0, y: 0, z: 0 };
};

/** Angle between two vectors, in degrees (0 = same direction). */
export function angleBetween(a: Point3, b: Point3): number {
  const la = length(a);
  const lb = length(b);
  if (la < 1e-9 || lb < 1e-9) return 0;
  const c = Math.min(1, Math.max(-1, dot(a, b) / (la * lb)));
  return (Math.acos(c) * 180) / Math.PI;
}

export function centroid(points: readonly Point3[]): Point3 {
  let x = 0;
  let y = 0;
  let z = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
    z += p.z;
  }
  const n = Math.max(points.length, 1);
  return { x: x / n, y: y / n, z: z / n };
}

/** Converts normalised image landmarks to an isotropic space (x scaled by aspect). */
export function toIsotropic(p: Point3, aspect: number): Point3 {
  return { x: p.x * aspect, y: p.y, z: p.z * aspect };
}

export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
/** Maps v from [a,b] to [0,1], clamped. */
export const ramp = (v: number, a: number, b: number) => clamp((v - a) / (b - a), 0, 1);
