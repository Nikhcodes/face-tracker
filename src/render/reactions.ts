/**
 * Reaction director: turns pose events into stickers, captions, particles and
 * camera effects, and draws them. Everything is drawn onto the same canvas as
 * the camera image so photos contain exactly what the preview shows.
 */
import { CHAOS_CAPTIONS, HELD_MESSAGES, type ParticleKind, type PoseDefinition, type StickerRef } from '../config/poses';
import { getPose } from '../config/registry';
import type { PoseId } from '../gestures/poseIds';
import type { HandObservation, Point3 } from '../gestures/types';
import { StickerLibrary, drawStickerCard, stickerAspect } from './stickers';
import { COLORS, FONTS, burstPath, easeOutBack, easeOutCubic, heartPath, roundRect, starPath } from './paint';
import { applyDesaturate } from './effects';

export interface View {
  w: number;
  h: number;
  mirror: boolean;
}

export interface ReactionFrame {
  now: number;
  hands: HandObservation[];
  view: View;
  reducedMotion: boolean;
  chaos: boolean;
  stickers: boolean;
  reactions: boolean;
}

interface Particle {
  kind: ParticleKind;
  x: number;
  y: number;
  vx: number;
  vy: number;
  rot: number;
  vr: number;
  size: number;
  color: string;
  born: number;
  life: number;
  gravity: number;
  drag: number;
}

interface Timed {
  start: number;
  exitAt: number | null;
}

const MAX_PARTICLES = 320;
const CAPTION_MIN_MS = 1100;
const EXIT_MS = 260;

export const toView = (p: Point3, v: View) => ({ x: (v.mirror ? 1 - p.x : p.x) * v.w, y: p.y * v.h });

function pick<T>(list: readonly T[], avoid?: T): T {
  if (list.length <= 1) return list[0];
  let item = list[Math.floor(Math.random() * list.length)];
  if (item === avoid) item = list[(list.indexOf(item) + 1) % list.length];
  return item;
}

export class ReactionDirector {
  private sticker: (Timed & { pose: PoseDefinition; ref: StickerRef }) | null = null;
  private caption: (Timed & { text: string; impact: boolean }) | null = null;
  private held: { text: string; start: number } | null = null;
  private particles: Particle[] = [];
  private active: PoseDefinition | null = null;
  private activeSince = 0;
  private shake = { until: 0, amp: 0 };
  private desaturateUntil = 0;
  private spotlight = 0;
  private letterbox = 0;
  private lastCaption = '';
  private lastVariant = new Map<PoseId, StickerRef>();
  private lastTrickle = 0;
  private muzzleAt = 0;
  private flashAt = -Infinity;

  constructor(private readonly lib: StickerLibrary) {}

  get activePose(): PoseId | null {
    return this.active?.id ?? null;
  }

  reset(): void {
    this.sticker = null;
    this.caption = null;
    this.held = null;
    this.particles = [];
    this.active = null;
    this.spotlight = 0;
    this.letterbox = 0;
  }

  /** Camera-flash white-out used by photo capture. */
  flash(now: number): void {
    this.flashAt = now;
  }

  enter(id: PoseId, f: ReactionFrame): void {
    const pose = getPose(id);
    this.active = pose;
    this.activeSince = f.now;
    const variant = pick(pose.stickers, this.lastVariant.get(id));
    this.lastVariant.set(id, variant);
    if (variant) this.sticker = { pose, ref: variant, start: f.now, exitAt: null };

    if (pose.captions.length || f.chaos) {
      const text = f.chaos && Math.random() < 0.6 ? pick(CHAOS_CAPTIONS, this.lastCaption) : pick(pose.captions, this.lastCaption);
      if (text) {
        this.lastCaption = text;
        this.caption = { text, start: f.now, exitAt: null, impact: !!pose.reaction.impact || (f.chaos && Math.random() < 0.4) };
      }
    } else {
      this.caption = null;
    }

    const r = pose.reaction;
    const chaosBoost = f.chaos ? 1.6 : 1;
    if (r.particles) this.burst(r.particles.kind, Math.round(r.particles.count * chaosBoost), r.particles.anchor, f);
    if ((r.shake || f.chaos) && !f.reducedMotion) this.shake = { until: f.now + 420, amp: (r.shake ?? 4) * (f.view.h / 720) };
    if (r.desaturate) this.desaturateUntil = f.now + 1600;
    if (r.muzzle) this.muzzleAt = f.now;
  }

  exit(id: PoseId, f: ReactionFrame): void {
    if (this.active?.id !== id) return;
    this.active = null;
    if (this.sticker && this.sticker.exitAt === null) this.sticker.exitAt = f.now;
    if (this.caption && this.caption.exitAt === null) {
      this.caption.exitAt = Math.max(f.now, this.caption.start + CAPTION_MIN_MS);
    }
  }

  heldFor(id: PoseId, f: ReactionFrame): void {
    if (this.active?.id !== id || id === 'idle') return;
    this.held = { text: pick(HELD_MESSAGES), start: f.now };
    if (f.reactions) this.burst('sparkles', f.reducedMotion ? 4 : 10, 'hand', f);
  }

  private anchor(kind: 'hand' | 'screen' | 'top', f: ReactionFrame): () => { x: number; y: number } {
    const { w, h } = f.view;
    if (kind === 'top') return () => ({ x: Math.random() * w, y: -20 - Math.random() * h * 0.2 });
    if (kind === 'screen' || !f.hands.length) return () => ({ x: w * (0.2 + Math.random() * 0.6), y: h * (0.25 + Math.random() * 0.4) });
    const pts = f.hands.map((hand) => toView(hand.image[9], f.view));
    return () => {
      const p = pts[Math.floor(Math.random() * pts.length)];
      return { x: p.x + (Math.random() - 0.5) * w * 0.08, y: p.y + (Math.random() - 0.5) * h * 0.08 };
    };
  }

  private burst(kind: ParticleKind, count: number, where: 'hand' | 'screen' | 'top', f: ReactionFrame): void {
    if (!f.reactions) return;
    const n = Math.round(count * (f.reducedMotion ? 0.25 : 1));
    const at = this.anchor(where, f);
    const s = f.view.h / 720;
    for (let i = 0; i < n && this.particles.length < MAX_PARTICLES; i++) {
      const p = at();
      const base: Particle = {
        kind,
        x: p.x,
        y: p.y,
        vx: (Math.random() - 0.5) * 260 * s,
        vy: (-120 - Math.random() * 220) * s,
        rot: Math.random() * Math.PI * 2,
        vr: (Math.random() - 0.5) * 6,
        size: (10 + Math.random() * 12) * s,
        color: COLORS.accent,
        born: f.now,
        life: 1100 + Math.random() * 900,
        gravity: 380 * s,
        drag: 0.9,
      };
      if (kind === 'confetti') {
        Object.assign(base, {
          vx: (Math.random() - 0.5) * 160 * s,
          vy: (60 + Math.random() * 160) * s,
          gravity: 140 * s,
          size: (8 + Math.random() * 10) * s,
          color: pick(COLORS.confetti),
          life: 2600 + Math.random() * 1600,
          vr: (Math.random() - 0.5) * 10,
        });
      } else if (kind === 'rain') {
        Object.assign(base, {
          vx: -30 * s,
          vy: (520 + Math.random() * 240) * s,
          gravity: 300 * s,
          size: (14 + Math.random() * 16) * s,
          color: 'rgba(170,200,255,0.7)',
          life: 1600,
          vr: 0,
        });
      } else if (kind === 'hearts') {
        Object.assign(base, { vy: (-60 - Math.random() * 120) * s, gravity: -40 * s, color: COLORS.heart, size: (14 + Math.random() * 16) * s, life: 1800 + Math.random() * 900 });
      } else if (kind === 'stars') {
        Object.assign(base, { color: Math.random() < 0.7 ? COLORS.accent : COLORS.paper, gravity: 260 * s });
      } else if (kind === 'notes') {
        Object.assign(base, { vy: (-90 - Math.random() * 120) * s, gravity: -30 * s, color: Math.random() < 0.5 ? COLORS.accent : COLORS.paper, size: (16 + Math.random() * 12) * s, vr: (Math.random() - 0.5) * 2 });
      } else {
        // sparkles
        Object.assign(base, { vx: (Math.random() - 0.5) * 140 * s, vy: (-40 - Math.random() * 120) * s, gravity: 40 * s, color: Math.random() < 0.6 ? COLORS.paper : COLORS.accent, life: 700 + Math.random() * 700 });
      }
      this.particles.push(base);
    }
  }

  update(dtMs: number, f: ReactionFrame): void {
    const dt = Math.min(dtMs, 64) / 1000;
    const alive: Particle[] = [];
    for (const p of this.particles) {
      if (f.now - p.born > p.life) continue;
      p.vy += p.gravity * dt;
      p.vx *= Math.pow(p.drag, dt * 4);
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.rot += p.vr * dt;
      if (p.y < f.view.h + 60 && p.y > -f.view.h) alive.push(p);
    }
    this.particles = alive;

    const r = this.active?.reaction;
    // gentle trickle while sparkle-type poses are held
    if (r?.particles && r.particles.anchor === 'hand' && f.now - this.lastTrickle > (f.reducedMotion ? 700 : 260)) {
      this.lastTrickle = f.now;
      this.burst(r.particles.kind, 1, 'hand', f);
    }
    if (r?.muzzle && f.now - this.muzzleAt > 1400) this.muzzleAt = f.now;
    const target = (on: boolean | undefined) => (on && f.reactions ? 1 : 0);
    const k = 1 - Math.pow(0.001, dt * (f.reducedMotion ? 4 : 2.5));
    this.spotlight += (target(r?.spotlight) - this.spotlight) * k;
    this.letterbox += (target(r?.letterbox) - this.letterbox) * k;

    if (this.sticker?.exitAt !== null && this.sticker && f.now - this.sticker.exitAt > EXIT_MS) this.sticker = null;
    if (this.caption?.exitAt !== null && this.caption && f.now - this.caption.exitAt > EXIT_MS) this.caption = null;
    if (this.held && f.now - this.held.start > 1400) this.held = null;
  }

  /** Whole-frame offset for screen shake. */
  shakeOffset(now: number): { x: number; y: number; scale: number } {
    if (now >= this.shake.until) return { x: 0, y: 0, scale: 1 };
    const k = (this.shake.until - now) / 420;
    const a = this.shake.amp * k;
    return { x: (Math.random() - 0.5) * 2 * a, y: (Math.random() - 0.5) * 2 * a, scale: 1 + 0.02 * k };
  }

  /** Effects that change the image itself (drawn before vignette/grain). */
  drawImageEffects(ctx: CanvasRenderingContext2D, f: ReactionFrame): void {
    const { w, h } = f.view;
    if (!f.reactions) return;
    const desat = Math.max(0, (this.desaturateUntil - f.now) / 1600);
    applyDesaturate(ctx, w, h, easeOutCubic(desat) * 0.85);
    if (this.spotlight > 0.01) {
      const c = f.hands.length ? toView(f.hands[0].image[9], f.view) : { x: w / 2, y: h * 0.45 };
      const cx = f.hands.length > 1 ? (c.x + toView(f.hands[1].image[9], f.view).x) / 2 : c.x;
      const g = ctx.createRadialGradient(cx, h * 0.45, h * 0.15, cx, h * 0.45, h * 0.85);
      g.addColorStop(0, 'rgba(0,0,0,0)');
      g.addColorStop(1, `rgba(0,0,0,${0.6 * this.spotlight})`);
      ctx.save();
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, w, h);
      ctx.restore();
    }
  }

  /** Overlays drawn on top of the image. */
  drawOverlays(ctx: CanvasRenderingContext2D, f: ReactionFrame): void {
    const { w, h } = f.view;
    const unit = Math.min(w, h);
    if (f.reactions) {
      this.drawAnchored(ctx, f, unit);
      this.drawParticles(ctx, f.now);
      if (this.letterbox > 0.01) {
        const bar = h * 0.1 * easeOutCubic(this.letterbox);
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, w, bar);
        ctx.fillRect(0, h - bar, w, bar);
      }
    }
    if (f.stickers && this.sticker) this.drawSticker(ctx, f, unit);
    if (f.reactions && this.caption) this.drawCaption(ctx, f, unit);
    if (f.reactions && this.held) this.drawHeld(ctx, f, unit);
    const flashAge = f.now - this.flashAt;
    if (flashAge < 360) {
      ctx.fillStyle = `rgba(255,255,255,${0.85 * (1 - flashAge / 360)})`;
      ctx.fillRect(0, 0, w, h);
    }
  }

  private drawAnchored(ctx: CanvasRenderingContext2D, f: ReactionFrame, unit: number): void {
    const r = this.active?.reaction;
    if (!r || !f.hands.length) return;
    const hand = f.hands[0];
    const t = f.now - this.activeSince;
    if (r.reticle) {
      const tip = toView(hand.image[8], f.view);
      const rad = unit * 0.06 * (0.9 + 0.1 * Math.sin(t / 160));
      const appear = easeOutBack(t / 300);
      ctx.save();
      ctx.translate(tip.x, tip.y);
      ctx.rotate(f.reducedMotion ? 0 : t / 900);
      ctx.scale(appear, appear);
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = unit * 0.006;
      ctx.beginPath();
      ctx.arc(0, 0, rad, 0, Math.PI * 2);
      ctx.stroke();
      for (let i = 0; i < 4; i++) {
        ctx.rotate(Math.PI / 2);
        ctx.beginPath();
        ctx.moveTo(rad * 0.55, 0);
        ctx.lineTo(rad * 1.45, 0);
        ctx.stroke();
      }
      ctx.restore();
    }
    if (r.muzzle) {
      const age = f.now - this.muzzleAt;
      if (age < 260) {
        const tip = toView(hand.image[8], f.view);
        const k = 1 - age / 260;
        ctx.save();
        ctx.translate(tip.x, tip.y);
        ctx.globalAlpha = k;
        ctx.fillStyle = COLORS.accent;
        starPath(ctx, 8, unit * 0.075 * (0.6 + k * 0.6), unit * 0.03);
        ctx.fill();
        ctx.fillStyle = COLORS.paper;
        starPath(ctx, 8, unit * 0.035, unit * 0.015);
        ctx.fill();
        ctx.restore();
      }
    }
    if (r.thought) {
      const c = toView(hand.image[9], f.view);
      const bx = c.x + unit * 0.12;
      const by = Math.max(unit * 0.14, c.y - unit * 0.3);
      const appear = easeOutBack(t / 350);
      ctx.save();
      ctx.fillStyle = 'rgba(246,243,236,0.95)';
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = unit * 0.004;
      const dots = [
        [c.x + unit * 0.04, c.y - unit * 0.1, unit * 0.012],
        [c.x + unit * 0.07, c.y - unit * 0.17, unit * 0.02],
      ];
      for (const [x, y, rr] of dots) {
        ctx.beginPath();
        ctx.arc(x, y, rr * appear, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.translate(bx, by);
      ctx.scale(appear, appear);
      ctx.beginPath();
      ctx.ellipse(0, 0, unit * 0.11, unit * 0.07, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = COLORS.ink;
      for (let i = 0; i < 3; i++) {
        const bounce = f.reducedMotion ? 0 : Math.sin(t / 180 - i * 0.8) * unit * 0.008;
        ctx.beginPath();
        ctx.arc((i - 1) * unit * 0.04, bounce, unit * 0.011, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  }

  private drawParticles(ctx: CanvasRenderingContext2D, now: number): void {
    for (const p of this.particles) {
      const age = (now - p.born) / p.life;
      const alpha = age > 0.75 ? 1 - (age - 0.75) / 0.25 : 1;
      ctx.save();
      ctx.globalAlpha = Math.max(0, alpha);
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.fillStyle = p.color;
      ctx.strokeStyle = p.color;
      switch (p.kind) {
        case 'confetti':
          ctx.scale(1, Math.abs(Math.cos(p.rot * 2)) * 0.8 + 0.2);
          ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
          break;
        case 'rain':
          ctx.rotate(-p.rot);
          ctx.lineWidth = Math.max(1, p.size * 0.12);
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(-p.size * 0.1, -p.size);
          ctx.stroke();
          break;
        case 'hearts':
          ctx.rotate(-p.rot + Math.sin(p.rot) * 0.3);
          heartPath(ctx, p.size);
          ctx.fill();
          break;
        case 'stars':
          starPath(ctx, 5, p.size * 0.6, p.size * 0.26);
          ctx.fill();
          break;
        case 'notes':
          ctx.rotate(-p.rot + Math.sin(p.rot) * 0.25);
          ctx.beginPath();
          ctx.ellipse(0, 0, p.size * 0.32, p.size * 0.24, -0.4, 0, Math.PI * 2);
          ctx.fill();
          ctx.fillRect(p.size * 0.22, -p.size * 0.95, p.size * 0.09, p.size * 0.95);
          ctx.fillRect(p.size * 0.22, -p.size * 0.95, p.size * 0.38, p.size * 0.12);
          break;
        default:
          starPath(ctx, 4, p.size * 0.55, p.size * 0.12);
          ctx.fill();
      }
      ctx.restore();
    }
  }

  private drawSticker(ctx: CanvasRenderingContext2D, f: ReactionFrame, unit: number): void {
    const s = this.sticker;
    if (!s) return;
    const { w, h } = f.view;
    const aspect = stickerAspect(s.ref);
    const isIdle = s.pose.id === 'idle';
    const cardW = Math.min(w * (isIdle ? 0.18 : 0.26), h * (isIdle ? 0.24 : 0.36) * aspect, unit * 0.5);
    const cardH = cardW / aspect;
    const margin = unit * 0.05;
    const cx = margin + cardW / 2;
    const cy = margin + cardH / 2;
    const t = f.now - s.start;
    let k = f.reducedMotion ? Math.min(1, t / 150) : easeOutBack(t / 380);
    if (s.exitAt !== null) k *= 1 - easeOutCubic((f.now - s.exitAt) / EXIT_MS);
    if (k <= 0.01) return;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(((f.reducedMotion ? -4 : -9 + 5 * easeOutCubic(t / 500)) * Math.PI) / 180);
    ctx.scale(k, k);
    ctx.globalAlpha = Math.min(1, k * 1.5) * (isIdle ? 0.92 : 1);
    drawStickerCard(ctx, this.lib, s.pose, s.ref, 0, 0, cardW, t);
    ctx.restore();
  }

  private drawCaption(ctx: CanvasRenderingContext2D, f: ReactionFrame, unit: number): void {
    const c = this.caption;
    if (!c) return;
    const { w, h } = f.view;
    const t = f.now - c.start;
    let k = f.reducedMotion ? Math.min(1, t / 160) : easeOutBack(t / 420);
    let alpha = Math.min(1, t / 120);
    if (c.exitAt !== null && f.now > c.exitAt) {
      const e = easeOutCubic((f.now - c.exitAt) / EXIT_MS);
      alpha *= 1 - e;
      k *= 1 - e * 0.15;
    }
    if (alpha <= 0.01) return;
    let size = unit * 0.085;
    ctx.save();
    ctx.font = `800 ${size}px ${FONTS.display}`;
    let tw = ctx.measureText(c.text).width;
    const maxW = w * 0.84;
    if (tw > maxW) {
      size *= maxW / tw;
      ctx.font = `800 ${size}px ${FONTS.display}`;
      tw = ctx.measureText(c.text).width;
    }
    const letterboxBar = h * 0.1 * this.letterbox;
    const y = h - letterboxBar - unit * 0.06 - size * 0.7;
    ctx.translate(w / 2, y);
    ctx.rotate((-2.2 * Math.PI) / 180);
    ctx.scale(k, k);
    ctx.globalAlpha = alpha;
    if (c.impact) {
      ctx.save();
      ctx.fillStyle = COLORS.paper;
      ctx.strokeStyle = COLORS.ink;
      ctx.lineWidth = unit * 0.008;
      burstPath(ctx, tw * 0.68 + size, size * 1.6, 14, c.text.length);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
    const padX = size * 0.45;
    const padY = size * 0.28;
    ctx.fillStyle = COLORS.accent;
    roundRect(ctx, -tw / 2 - padX, -size * 0.62 - padY, tw + padX * 2, size * 1.24 + padY * 2, size * 0.16);
    ctx.fill();
    ctx.fillStyle = COLORS.accentInk;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(c.text, 0, size * 0.04);
    ctx.restore();
  }

  private drawHeld(ctx: CanvasRenderingContext2D, f: ReactionFrame, unit: number): void {
    const held = this.held;
    if (!held) return;
    const t = f.now - held.start;
    const alpha = t < 150 ? t / 150 : t > 1100 ? Math.max(0, 1 - (t - 1100) / 300) : 1;
    const size = unit * 0.034;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = `700 ${size}px ${FONTS.ui}`;
    const tw = ctx.measureText(held.text).width;
    const x = f.view.w - unit * 0.05 - tw / 2 - size;
    const y = unit * 0.06 + size;
    const lift = f.reducedMotion ? 0 : (1 - easeOutCubic(t / 300)) * size;
    ctx.translate(x, y + lift);
    ctx.fillStyle = 'rgba(12,12,13,0.78)';
    roundRect(ctx, -tw / 2 - size, -size, tw + size * 2, size * 2, size);
    ctx.fill();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = Math.max(1, size * 0.08);
    ctx.stroke();
    ctx.fillStyle = COLORS.accent;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(held.text, 0, size * 0.05);
    ctx.restore();
  }
}
