/** Live preview of a pose's reaction, drawn with the same code the camera uses. */
import { useEffect, useRef } from 'react';
import type { PoseDefinition } from '../../config/poses';
import { glyphGeometry, paintGlyph } from '../../render/glyph';
import { COLORS, drawTape, easeOutBack, easeOutCubic, starPath, heartPath } from '../../render/paint';
import { StickerLibrary, drawStickerCard, stickerAspect } from '../../render/stickers';
import { prefersReducedMotion, settingsStore } from '../../state/settings';

const CYCLE_MS = 2600;
const lib = new StickerLibrary();

export function PosePreview({ pose }: { pose: PoseDefinition }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const poseRef = useRef(pose);
  const startRef = useRef(performance.now());

  useEffect(() => {
    const changedShape = poseRef.current.id !== pose.id;
    poseRef.current = pose;
    if (changedShape) startRef.current = performance.now();
  }, [pose]);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const reduced = prefersReducedMotion(settingsStore.get());
    let raf = 0;
    const resize = () => {
      const r = canvas.getBoundingClientRect();
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
    };
    const draw = (now: number) => {
      const p = poseRef.current;
      const w = canvas.width;
      const h = canvas.height;
      const unit = Math.min(w, h);
      ctx.clearRect(0, 0, w, h);
      const t = reduced ? 900 : Math.max(0, now - startRef.current);
      const cycle = Math.floor(t / CYCLE_MS);
      const local = t % CYCLE_MS;

      if (p.glyph.length) {
        const size = unit * 0.66;
        paintGlyph(ctx, glyphGeometry(p.glyph, 0.06, false), w * 0.62 - size / 2, h * 0.46 - size / 2, size, COLORS.paper, '#1a1a17');
      }

      // a few static particles hint at the chosen effect
      const kind = p.reaction.particles?.kind;
      if (kind) {
        for (let i = 0; i < 9; i++) {
          const x = w * (0.45 + 0.4 * Math.abs(Math.sin(i * 12.9)));
          const y = h * (0.16 + 0.55 * Math.abs(Math.sin(i * 7.3)));
          const s = unit * (0.028 + 0.012 * (i % 3));
          ctx.save();
          ctx.translate(x, y + (reduced ? 0 : Math.sin(now / 500 + i) * unit * 0.01));
          ctx.globalAlpha = 0.85;
          ctx.fillStyle = kind === 'hearts' ? COLORS.heart : kind === 'confetti' ? COLORS.confetti[i % COLORS.confetti.length] : i % 2 ? COLORS.accent : COLORS.paper;
          if (kind === 'hearts') {
            heartPath(ctx, s * 1.4);
            ctx.fill();
          } else if (kind === 'confetti') {
            ctx.rotate(i);
            ctx.fillRect(-s / 2, -s / 4, s, s / 2);
          } else if (kind === 'rain') {
            ctx.fillStyle = 'rgba(170,200,255,0.75)';
            ctx.fillRect(0, 0, Math.max(1, s * 0.12), s * 1.4);
          } else {
            starPath(ctx, kind === 'stars' ? 5 : 4, s * 0.6, s * (kind === 'stars' ? 0.26 : 0.13));
            ctx.fill();
          }
          ctx.restore();
        }
      }

      const ref0 = p.stickers.length ? p.stickers[cycle % p.stickers.length] : null;
      if (ref0) {
        const k = reduced ? 1 : easeOutBack(local / 380) * (local > CYCLE_MS - 260 && p.stickers.length > 1 ? 1 - easeOutCubic((local - (CYCLE_MS - 260)) / 240) : 1);
        const cw = Math.min(w * 0.34, unit * 0.44);
        const ch = cw / stickerAspect(ref0);
        const m = unit * 0.07;
        ctx.save();
        ctx.translate(m + cw / 2, m + ch / 2);
        ctx.rotate((-6 * Math.PI) / 180);
        ctx.scale(k, k);
        drawStickerCard(ctx, lib, p, ref0, 0, 0, cw, local);
        ctx.restore();
      }

      const caption = p.captions.length ? p.captions[cycle % p.captions.length] : null;
      if (caption) {
        const k = reduced ? 1 : easeOutBack((local - 120) / 420);
        if (k > 0) {
          ctx.save();
          ctx.translate(w / 2, h - unit * 0.13);
          ctx.rotate((-2.2 * Math.PI) / 180);
          ctx.scale(k, k);
          drawTape(ctx, caption, unit * 0.075, w * 0.84);
          ctx.restore();
        }
      }
    };
    const loop = (now: number) => {
      draw(now);
      raf = requestAnimationFrame(loop);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    void Promise.all([lib.load(), document.fonts?.ready]);
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  return (
    <div className="ed-preview">
      <canvas ref={ref} role="img" aria-label={`Preview of the ${pose.name} reaction`} />
    </div>
  );
}
