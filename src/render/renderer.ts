/**
 * Per-frame compositor. Draw order:
 *   camera (mirrored) → colour look → reaction image effects → vignette → grain
 *   → landmarks → reactions/stickers/captions → countdown → frame.
 */
import { describeFeatures, computeFeatures } from '../gestures/features';
import { HAND_CONNECTIONS, type HandObservation } from '../gestures/types';
import type { Quality, Settings } from '../state/settings';
import { Grain, Vignette, applyFilter } from './effects';
import { COLORS, FONTS, easeOutBack } from './paint';
import { ReactionDirector, toView, type ReactionFrame, type View } from './reactions';

export const CANVAS_LONG_SIDE: Record<Quality, number> = { performance: 960, balanced: 1280, high: 1920 };

export interface RenderInput {
  now: number;
  dt: number;
  video: HTMLVideoElement;
  settings: Settings;
  hands: HandObservation[];
  aspect: number;
  reducedMotion: boolean;
  countdown: { endsAt: number; seconds: number } | null;
  goldFrame: boolean;
}

/** Computes the canvas size for a video frame, keeping its aspect ratio. */
export function canvasSizeFor(videoW: number, videoH: number, quality: Quality): { width: number; height: number } {
  if (!videoW || !videoH) return { width: 1280, height: 720 };
  const k = Math.min(1, CANVAS_LONG_SIDE[quality] / Math.max(videoW, videoH));
  return { width: Math.round(videoW * k), height: Math.round(videoH * k) };
}

export class StudioRenderer {
  private readonly ctx: CanvasRenderingContext2D;
  private readonly vignette = new Vignette();
  private readonly grain = new Grain();

  constructor(
    readonly canvas: HTMLCanvasElement,
    readonly director: ReactionDirector,
  ) {
    const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
    if (!ctx) throw new Error('Canvas 2D is not available');
    this.ctx = ctx;
  }

  private resize(input: RenderInput): View {
    const { width, height } = canvasSizeFor(input.video.videoWidth, input.video.videoHeight, input.settings.quality);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }
    return { w: width, h: height, mirror: input.settings.mirror };
  }

  render(input: RenderInput): void {
    const { ctx } = this;
    const view = this.resize(input);
    const { w, h } = view;
    const s = input.settings;
    const frame: ReactionFrame = {
      now: input.now,
      hands: input.hands,
      view,
      reducedMotion: input.reducedMotion,
      chaos: s.chaos,
      stickers: s.stickers,
      reactions: s.reactions,
    };
    this.director.update(input.dt, frame);

    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    const shake = this.director.shakeOffset(input.now);
    if (shake.scale !== 1) {
      ctx.translate(w / 2 + shake.x, h / 2 + shake.y);
      ctx.scale(shake.scale, shake.scale);
      ctx.translate(-w / 2, -h / 2);
    }
    if (view.mirror) {
      ctx.translate(w, 0);
      ctx.scale(-1, 1);
    }
    if (input.video.readyState >= 2) ctx.drawImage(input.video, 0, 0, w, h);
    ctx.restore();

    applyFilter(ctx, w, h, s.filter);
    this.director.drawImageEffects(ctx, frame);
    if (s.vignette || s.filter === 'cinematic') this.vignette.draw(ctx, w, h, s.filter === 'cinematic' ? 1.25 : 0.85);
    if (s.grain) this.grain.draw(ctx, w, h, !input.reducedMotion);
    if (s.landmarks || s.debug) this.drawLandmarks(input, view);
    this.director.drawOverlays(ctx, frame);
    if (input.countdown) this.drawCountdown(input, view);
    if (input.goldFrame) this.drawGoldFrame(view);
  }

  private drawLandmarks(input: RenderInput, view: View): void {
    const { ctx } = this;
    const unit = Math.min(view.w, view.h);
    for (const hand of input.hands) {
      const pts = hand.image.map((p) => toView(p, view));
      ctx.save();
      ctx.lineCap = 'round';
      ctx.strokeStyle = 'rgba(246,243,236,0.85)';
      ctx.lineWidth = Math.max(1.5, unit * 0.004);
      ctx.beginPath();
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.moveTo(pts[a].x, pts[a].y);
        ctx.lineTo(pts[b].x, pts[b].y);
      }
      ctx.stroke();
      pts.forEach((p, i) => {
        const tip = i === 4 || i === 8 || i === 12 || i === 16 || i === 20;
        ctx.beginPath();
        ctx.arc(p.x, p.y, unit * (tip ? 0.009 : 0.006), 0, Math.PI * 2);
        ctx.fillStyle = tip ? COLORS.accent : COLORS.paper;
        ctx.fill();
        ctx.lineWidth = Math.max(1, unit * 0.002);
        ctx.strokeStyle = COLORS.ink;
        ctx.stroke();
      });
      if (input.settings.debug) {
        const f = computeFeatures(hand, input.aspect);
        const label = `${describeFeatures(f)}  ${hand.handedness[0]}${Math.round(hand.score * 100)}`;
        const size = Math.max(11, unit * 0.022);
        ctx.font = `600 ${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        const wrist = pts[0];
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(wrist.x - tw / 2 - 6, wrist.y + size * 0.6, tw + 12, size * 1.6);
        ctx.fillStyle = COLORS.accent;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, wrist.x, wrist.y + size * 1.4);
      }
      ctx.restore();
    }
  }

  private drawCountdown(input: RenderInput, view: View): void {
    const cd = input.countdown;
    if (!cd) return;
    const { ctx } = this;
    const remaining = Math.max(0, cd.endsAt - input.now);
    const n = Math.ceil(remaining / 1000);
    if (n <= 0) return;
    const within = 1 - (remaining % 1000) / 1000;
    const unit = Math.min(view.w, view.h);
    const r = unit * 0.13;
    const k = input.reducedMotion ? 1 : easeOutBack(within * 2.5);
    ctx.save();
    ctx.translate(view.w / 2, view.h / 2);
    ctx.fillStyle = 'rgba(12,12,13,0.55)';
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = unit * 0.01;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.arc(0, 0, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * (remaining / (cd.seconds * 1000)));
    ctx.stroke();
    ctx.scale(k, k);
    ctx.fillStyle = COLORS.paper;
    ctx.font = `800 ${r * 1.15}px ${FONTS.display}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(n), 0, r * 0.06);
    ctx.restore();
  }

  private drawGoldFrame(view: View): void {
    const { ctx } = this;
    const unit = Math.min(view.w, view.h);
    const b = unit * 0.018;
    ctx.save();
    const g = ctx.createLinearGradient(0, 0, view.w, view.h);
    g.addColorStop(0, '#f7e08b');
    g.addColorStop(0.5, COLORS.gold);
    g.addColorStop(1, '#a8812a');
    ctx.strokeStyle = g;
    ctx.lineWidth = b;
    ctx.strokeRect(b / 2, b / 2, view.w - b, view.h - b);
    ctx.fillStyle = g;
    const size = unit * 0.026;
    ctx.font = `800 ${size}px ${FONTS.display}`;
    ctx.textBaseline = 'bottom';
    ctx.fillText('CERTIFIED LEGEND', b * 2, view.h - b * 1.6);
    ctx.restore();
  }
}
