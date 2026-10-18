/**
 * Landing-page demo animation. It uses the app's real drawing code (hand
 * glyphs, sticker cards, caption tape) but no camera — and says so.
 */
import { useEffect, useRef } from 'react';
import { BUILTIN_BY_ID as POSE_BY_ID, type GlyphHand } from '../config/poses';
import type { PoseId } from '../gestures/poseIds';
import type { FingerPose, HandSpec, ThumbPose } from '../gestures/synthetic';
import { glyphGeometry, paintGlyph } from '../render/glyph';
import { COLORS, drawTape, easeOutBack, easeOutCubic } from '../render/paint';
import { StickerLibrary, drawStickerCard, stickerAspect } from '../render/stickers';
import { prefersReducedMotion, settingsStore } from '../state/settings';

const SEQUENCE: { id: PoseId; caption: string }[] = [
  { id: 'peace', caption: 'PLEASE REMAIN ICONIC.' },
  { id: 'rockon', caption: 'TOO MUCH AURA.' },
  { id: 'thumbsup', caption: 'THE COUNCIL APPROVES.' },
  { id: 'hello', caption: 'HELLO THERE.' },
  { id: 'point', caption: 'MAIN CHARACTER DETECTED.' },
];
const SEG_MS = 2900;
const MORPH_MS = 520;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const lerpFinger = (a: FingerPose, b: FingerPose, t: number): FingerPose => ({
  mcp: lerp(a.mcp, b.mcp, t),
  pip: lerp(a.pip, b.pip, t),
  dip: lerp(a.dip, b.dip, t),
  spread: lerp(a.spread ?? 0, b.spread ?? 0, t),
});
const lerpThumb = (a: ThumbPose, b: ThumbPose, t: number): ThumbPose => ({
  cmc: lerp(a.cmc, b.cmc, t),
  mcp: lerp(a.mcp, b.mcp, t),
  ip: lerp(a.ip, b.ip, t),
  abduct: lerp(a.abduct ?? 0, b.abduct ?? 0, t),
});
function lerpHand(a: GlyphHand, b: GlyphHand, t: number): GlyphHand {
  const sa = a.spec;
  const sb = b.spec;
  const spec: HandSpec = {
    index: lerpFinger(sa.index, sb.index, t),
    middle: lerpFinger(sa.middle, sb.middle, t),
    ring: lerpFinger(sa.ring, sb.ring, t),
    pinky: lerpFinger(sa.pinky, sb.pinky, t),
    thumb: lerpThumb(sa.thumb, sb.thumb, t),
  };
  return { spec, roll: lerp(a.roll ?? 0, b.roll ?? 0, t) };
}

export function HeroPreview() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const lib = new StickerLibrary();
    let raf = 0;
    let visible = true;
    let start = performance.now();
    const reduced = prefersReducedMotion(settingsStore.get());

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
    };

    const draw = (now: number) => {
      const w = canvas.width;
      const h = canvas.height;
      const unit = Math.min(w, h);
      ctx.clearRect(0, 0, w, h);
      const elapsed = reduced ? 1400 : Math.max(0, now - start);
      const seg = Math.floor(elapsed / SEG_MS) % SEQUENCE.length;
      const t = elapsed % SEG_MS;
      const cur = SEQUENCE[seg];
      const prev = SEQUENCE[(seg + SEQUENCE.length - 1) % SEQUENCE.length];
      const morph = easeOutCubic(t / MORPH_MS);
      const hand = lerpHand(POSE_BY_ID[prev.id].glyph[0], POSE_BY_ID[cur.id].glyph[0], morph);

      // viewfinder corner marks
      ctx.strokeStyle = 'rgba(246,243,236,0.28)';
      ctx.lineWidth = Math.max(1.5, unit * 0.004);
      const m = unit * 0.05;
      const L = unit * 0.07;
      for (const [x, y, dx, dy] of [
        [m, m, 1, 1],
        [w - m, m, -1, 1],
        [m, h - m, 1, -1],
        [w - m, h - m, -1, -1],
      ]) {
        ctx.beginPath();
        ctx.moveTo(x, y + dy * L);
        ctx.lineTo(x, y);
        ctx.lineTo(x + dx * L, y);
        ctx.stroke();
      }

      // the hand
      const geo = glyphGeometry([hand], 0.06, false);
      const size = unit * 0.62;
      const bob = reduced ? 0 : Math.sin(now / 700) * unit * 0.008;
      paintGlyph(ctx, geo, w * 0.6 - size / 2, h * 0.47 - size / 2 + bob, size, COLORS.paper, '#1a1a17');

      // sticker
      const pose = POSE_BY_ID[cur.id];
      const ref = pose.stickers[0];
      const st = t - 520;
      const leaving = t > SEG_MS - 300 ? easeOutCubic((t - (SEG_MS - 300)) / 260) : 0;
      if (st > 0 && ref) {
        const k = (reduced ? 1 : easeOutBack(st / 380)) * (1 - leaving);
        const cw = Math.min(w * 0.3, unit * 0.42);
        const ch = cw / stickerAspect(ref);
        ctx.save();
        ctx.translate(m + cw / 2 + unit * 0.03, m + ch / 2 + unit * 0.04);
        ctx.rotate(((-9 + 5 * easeOutCubic(st / 500)) * Math.PI) / 180);
        ctx.scale(k, k);
        drawStickerCard(ctx, lib, pose, ref, 0, 0, cw, st);
        ctx.restore();
      }
      // caption
      const ct = t - 680;
      if (ct > 0) {
        const k = reduced ? 1 : easeOutBack(ct / 420);
        ctx.save();
        ctx.globalAlpha = Math.min(1, ct / 120) * (1 - leaving);
        ctx.translate(w / 2, h - unit * 0.14);
        ctx.rotate((-2.2 * Math.PI) / 180);
        ctx.scale(k, k);
        drawTape(ctx, cur.caption, unit * 0.07, w * 0.8);
        ctx.restore();
      }
    };

    const loop = (now: number) => {
      draw(now);
      if (!reduced && visible && !document.hidden) raf = requestAnimationFrame(loop);
      else raf = 0;
    };
    const kick = () => {
      if (!raf) raf = requestAnimationFrame(loop);
    };

    resize();
    const ro = new ResizeObserver(() => {
      resize();
      kick();
    });
    ro.observe(canvas);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) kick();
    });
    io.observe(canvas);
    const onVis = () => !document.hidden && kick();
    document.addEventListener('visibilitychange', onVis);
    void Promise.all([lib.load(), document.fonts?.ready]).then(kick);
    start = performance.now();
    kick();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener('visibilitychange', onVis);
    };
  }, []);

  return (
    <figure className="preview-figure">
      <div className="preview">
        <canvas ref={canvasRef} role="img" aria-label="Animated preview: a hand changes pose and the camera answers with a sticker and a caption." />
      </div>
      <figcaption className="preview__tag">Preview animation. Your camera isn’t on.</figcaption>
    </figure>
  );
}
