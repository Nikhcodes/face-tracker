/**
 * 2D hand-silhouette geometry built from the synthetic hand model.
 * Shared by the SVG glyphs in the UI and the canvas stamps in the camera view.
 */
import type { GlyphHand } from '../config/poses';
import { synthHand } from '../gestures/synthetic';

export interface GlyphSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Stroke width in the same unit box. */
  w: number;
  /** Depth for painter's ordering (larger = further away). */
  z: number;
}

export interface GlyphGeometry {
  palms: Array<Array<[number, number]>>;
  segments: GlyphSegment[];
}

const FINGER_CHAINS = [
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20],
];
const PALM = [0, 1, 5, 9, 13, 17];
const cache = new Map<GlyphHand[], GlyphGeometry>();

/** Returns geometry fitted into a 0..1 box (aspect preserved, centred). */
export function glyphGeometry(hands: GlyphHand[], padding = 0.08, useCache = true): GlyphGeometry {
  const hit = useCache ? cache.get(hands) : undefined;
  if (hit) return hit;
  const raw = hands.map((h) => {
    const obs = synthHand(h.spec, {
      cx: 0.5 + (h.dx ?? 0),
      cy: 0.5 + (h.dy ?? 0),
      size: 1,
      aspect: 1,
      roll: h.roll ?? 0,
      yaw: h.yaw ?? 0,
      handedness: h.mirror ? 'Left' : 'Right',
    });
    return obs.image;
  });
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  const fingerW = 0.016;
  for (const pts of raw)
    for (const p of pts) {
      minX = Math.min(minX, p.x - fingerW);
      maxX = Math.max(maxX, p.x + fingerW);
      minY = Math.min(minY, p.y - fingerW);
      maxY = Math.max(maxY, p.y + fingerW);
    }
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const k = (1 - padding * 2) / span;
  const ox = (1 - (maxX - minX) * k) / 2;
  const oy = (1 - (maxY - minY) * k) / 2;
  const tx = (x: number) => ox + (x - minX) * k;
  const ty = (y: number) => oy + (y - minY) * k;

  const geometry: GlyphGeometry = { palms: [], segments: [] };
  for (const pts of raw) {
    geometry.palms.push(PALM.map((i) => [tx(pts[i].x), ty(pts[i].y)]));
    FINGER_CHAINS.forEach((chain, fi) => {
      for (let j = 0; j < chain.length - 1; j++) {
        const a = pts[chain[j]];
        const b = pts[chain[j + 1]];
        const width = (fi === 0 ? 0.03 : fi === 4 ? 0.024 : 0.027) * (1 - j * 0.07);
        geometry.segments.push({ x1: tx(a.x), y1: ty(a.y), x2: tx(b.x), y2: ty(b.y), w: width * k, z: (a.z + b.z) / 2 });
      }
    });
  }
  geometry.segments.sort((s1, s2) => s2.z - s1.z);
  if (useCache) cache.set(hands, geometry);
  return geometry;
}

/** Paints a glyph onto a canvas inside the box (x, y, size). */
export function paintGlyph(
  ctx: CanvasRenderingContext2D,
  geometry: GlyphGeometry,
  x: number,
  y: number,
  size: number,
  fill: string,
  outline: string,
): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.scale(size, size);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.fillStyle = fill;
  ctx.strokeStyle = fill;
  for (const palm of geometry.palms) {
    ctx.beginPath();
    palm.forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
    ctx.closePath();
    ctx.lineWidth = 0.05;
    ctx.fill();
    ctx.stroke();
  }
  const outlineW = 0.016;
  for (const s of geometry.segments) {
    ctx.strokeStyle = outline;
    ctx.lineWidth = s.w + outlineW;
    ctx.beginPath();
    ctx.moveTo(s.x1, s.y1);
    ctx.lineTo(s.x2, s.y2);
    ctx.stroke();
    ctx.strokeStyle = fill;
    ctx.lineWidth = s.w;
    ctx.stroke();
  }
  ctx.restore();
}
