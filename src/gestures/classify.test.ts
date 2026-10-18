import { describe, expect, it } from 'vitest';
import { classifyFrame, classifyHand, isHeart, type FrameResult } from './classify';
import { computeFeatures } from './features';
import { F_HALF, SPECS, T_REST, rollToPointThumb, synthHand, type HandSpec, type Placement } from './synthetic';
import type { FrameObservation, HandObservation } from './types';

const frame = (hands: HandObservation[], aspect = 16 / 9): FrameObservation => ({ hands, aspect });
const poseOf = (r: FrameResult) => (r.kind === 'pose' ? r.pose : `${r.kind}${r.kind === 'unknown' ? `:${r.reason}` : ''}`);

/** A broad sweep of orientations, sizes, aspect ratios, both hands and detector noise. */
function variations(): Placement[] {
  const out: Placement[] = [];
  let seed = 7;
  for (const handedness of ['Right', 'Left'] as const)
    for (const roll of [-60, -25, 0, 25, 60])
      for (const yaw of [-45, 0, 45])
        for (const pitch of [-25, 0, 25])
          for (const aspect of [16 / 9, 3 / 4]) {
            out.push({ handedness, roll, yaw, pitch, aspect, noise: 0.002, seed: seed++, size: 1.6 + (seed % 5) * 0.3 });
          }
  return out;
}

const SINGLE_CASES: Array<[string, HandSpec]> = [
  ['hello', SPECS.open],
  ['peace', SPECS.peace],
  ['point', SPECS.point],
  ['three', SPECS.three],
  ['rockon', SPECS.rockon],
  ['shaka', SPECS.shaka],
  ['fingergun', SPECS.gun],
];

describe('single-hand poses are orientation independent', () => {
  for (const [expected, spec] of SINGLE_CASES) {
    it(`${expected} — across ${variations().length} orientations`, () => {
      const misses: string[] = [];
      for (const p of variations()) {
        // keep the hand away from the "think" zone; irrelevant for these shapes anyway
        const got = poseOf(classifyFrame(frame([synthHand(spec, { ...p, cy: 0.35 })], p.aspect)));
        if (got !== expected) misses.push(`${JSON.stringify({ ...p, seed: undefined })} → ${got}`);
      }
      expect(misses).toEqual([]);
    });
  }

  it('recognises point even when the hand is sideways or upside down (old y-only logic could not)', () => {
    for (const roll of [90, -90, 180]) {
      expect(poseOf(classifyFrame(frame([synthHand(SPECS.point, { roll })])))).toBe('point');
      expect(poseOf(classifyFrame(frame([synthHand(SPECS.peace, { roll })])))).toBe('peace');
    }
  });

  it('treats a point with the thumb resting alongside as point, not finger gun', () => {
    expect(poseOf(classifyFrame(frame([synthHand({ ...SPECS.point, thumb: T_REST }, { cy: 0.3 })])))).toBe('point');
  });
});

describe('thumbs', () => {
  for (const target of ['up', 'down'] as const) {
    it(`thumbs ${target} in several yaw/pitch angles and both hands`, () => {
      for (const handedness of ['Right', 'Left'] as const)
        for (const turn of [-70, -35, 0, 35, 70])
          for (const tilt of [-20, 0, 20]) {
            const base: Placement = { handedness, cy: 0.35 };
            const roll = rollToPointThumb(SPECS.thumb, target, base);
            const p = { ...base, roll, turn, tilt, noise: 0.0015, seed: turn * 3 + tilt + 99 };
            const r = classifyFrame(frame([synthHand(SPECS.thumb, p)]));
            expect(poseOf(r), JSON.stringify(p)).toBe(target === 'up' ? 'thumbsup' : 'thumbsdown');
          }
    });
  }

  it('a sideways thumb is neither up nor down', () => {
    const base: Placement = { cy: 0.3 };
    const upRoll = rollToPointThumb(SPECS.thumb, 'up', base);
    const r = classifyFrame(frame([synthHand(SPECS.thumb, { ...base, roll: upRoll + 90 })]));
    expect(['thumbsup', 'thumbsdown']).not.toContain(poseOf(r));
  });
});

describe('position-dependent and two-hand poses', () => {
  it('fist near chin height → think; fist high in a corner → not a pose', () => {
    expect(poseOf(classifyFrame(frame([synthHand(SPECS.fist, { cx: 0.5, cy: 0.7 })])))).toBe('think');
    expect(poseOf(classifyFrame(frame([synthHand(SPECS.fist, { cx: 0.12, cy: 0.2 })])))).toBe('unknown:unrecognized');
  });

  it('two raised fists → rodrick; two low fists → not rodrick', () => {
    const high = [
      synthHand(SPECS.fist, { cx: 0.3, cy: 0.35, handedness: 'Right' }),
      synthHand(SPECS.fist, { cx: 0.7, cy: 0.35, handedness: 'Left' }),
    ];
    expect(poseOf(classifyFrame(frame(high)))).toBe('rodrick');
    const low = [
      synthHand(SPECS.fist, { cx: 0.3, cy: 0.8, handedness: 'Right' }),
      synthHand(SPECS.fist, { cx: 0.7, cy: 0.8, handedness: 'Left' }),
    ];
    expect(poseOf(classifyFrame(frame(low)))).not.toBe('rodrick');
  });

  it('two open hands → celebrate', () => {
    const r = classifyFrame(
      frame([
        synthHand(SPECS.open, { cx: 0.25, cy: 0.4, handedness: 'Right', roll: 15 }),
        synthHand(SPECS.open, { cx: 0.75, cy: 0.4, handedness: 'Left', roll: -15 }),
      ]),
    );
    expect(poseOf(r)).toBe('celebrate');
  });

  it('two thumbs up → doublethumbs', () => {
    const a: Placement = { cx: 0.3, cy: 0.4, handedness: 'Right' };
    const b: Placement = { cx: 0.7, cy: 0.4, handedness: 'Left' };
    const r = classifyFrame(
      frame([
        synthHand(SPECS.thumb, { ...a, roll: rollToPointThumb(SPECS.thumb, 'up', a) }),
        synthHand(SPECS.thumb, { ...b, roll: rollToPointThumb(SPECS.thumb, 'up', b) }),
      ]),
    );
    expect(poseOf(r)).toBe('doublethumbs');
  });

  it('heart geometry: touching index tips above touching thumb tips', () => {
    const aspect = 16 / 9;
    const make = (cx: number, h: 'Left' | 'Right') => synthHand(SPECS.heartHalf, { cx, cy: 0.5, handedness: h });
    const a = make(0.4, 'Right');
    const b = make(0.6, 'Left');
    // place the key points explicitly: index tips meet at the top, thumb tips meet below
    const set = (hand: HandObservation, i: number, x: number, y: number) => (hand.image[i] = { ...hand.image[i], x, y });
    set(a, 8, 0.495, 0.4);
    set(b, 8, 0.505, 0.4);
    set(a, 4, 0.497, 0.62);
    set(b, 4, 0.503, 0.62);
    set(a, 0, 0.36, 0.7);
    set(b, 0, 0.64, 0.7);
    const fa = computeFeatures(a, aspect);
    const fb = computeFeatures(b, aspect);
    expect(isHeart(a, b, aspect, fa, fb)).toBe(true);
    expect(poseOf(classifyFrame(frame([a, b], aspect)))).toBe('heart');
    // pull the index tips apart → no heart
    set(b, 8, 0.62, 0.4);
    expect(isHeart(a, b, aspect, fa, computeFeatures(b, aspect))).toBe(false);
  });
});

describe('unknown, ambiguous and empty frames', () => {
  it('no hands → none', () => {
    expect(classifyFrame(frame([]))).toEqual({ kind: 'none' });
  });

  it('ignores hands below the detector score threshold', () => {
    const weak = { ...synthHand(SPECS.peace), score: 0.2 };
    expect(classifyFrame(frame([weak]))).toEqual({ kind: 'none' });
  });

  it('a loosely curled hand is reported as uncertain, not as a random pose', () => {
    const half: HandSpec = { index: F_HALF, middle: F_HALF, ring: F_HALF, pinky: F_HALF, thumb: T_REST };
    expect(poseOf(classifyFrame(frame([synthHand(half, { cy: 0.3 })])))).toBe('unknown:uncertain');
  });

  it('two different, equally confident single-hand poses → ambiguous', () => {
    const r = classifyFrame(
      frame([
        synthHand(SPECS.peace, { cx: 0.3, cy: 0.35, handedness: 'Right' }),
        synthHand(SPECS.rockon, { cx: 0.7, cy: 0.35, handedness: 'Left' }),
      ]),
    );
    expect(poseOf(r)).toBe('unknown:ambiguous');
  });

  it('a pose on one hand plus an unrecognised other hand → the pose', () => {
    const half: HandSpec = { index: F_HALF, middle: F_HALF, ring: F_HALF, pinky: F_HALF, thumb: T_REST };
    const r = classifyFrame(
      frame([synthHand(SPECS.peace, { cx: 0.3, cy: 0.35 }), synthHand(half, { cx: 0.7, cy: 0.35, handedness: 'Left' })]),
    );
    expect(poseOf(r)).toBe('peace');
  });
});

describe('mirroring', () => {
  it('mirroring the frame (selfie view) does not change the verdict', () => {
    const mirror = (h: HandObservation): HandObservation => ({
      ...h,
      image: h.image.map((p) => ({ ...p, x: 1 - p.x })),
      world: h.world.map((p) => ({ ...p, x: -p.x })),
      handedness: h.handedness === 'Left' ? 'Right' : 'Left',
    });
    for (const [expected, spec] of SINGLE_CASES) {
      const hand = synthHand(spec, { roll: 20, yaw: 30, cy: 0.3 });
      expect(poseOf(classifyFrame(frame([mirror(hand)])))).toBe(expected);
    }
  });
});

describe('classifyHand', () => {
  it('reports per-digit features for the debug overlay', () => {
    const v = classifyHand(synthHand(SPECS.peace), 16 / 9);
    expect(v.shape).toBe('peace');
    expect(v.features.fingers.index.state).toBe('extended');
    expect(v.features.fingers.ring.state).toBe('folded');
    expect(v.confidence).toBeGreaterThan(0.8);
  });
});
