import { describe, expect, it } from 'vitest';
import type { FrameResult } from './classify';
import type { GesturePoseId } from './poseIds';
import { Cooldown, PoseStabilizer, type StabilizerEvent } from './stabilizer';

const pose = (p: GesturePoseId, confidence = 0.9): FrameResult => ({ kind: 'pose', pose: p, confidence, hands: [] });
const none: FrameResult = { kind: 'none' };
const unknown: FrameResult = { kind: 'unknown', reason: 'unrecognized', hands: [] };

/** Feeds `result` every 33 ms from `t0` for `ms`, returning the final state and all events. */
function feed(s: PoseStabilizer, result: FrameResult | ((t: number) => FrameResult), t0: number, ms: number) {
  const events: StabilizerEvent[] = [];
  let state = s.state();
  for (let t = t0; t <= t0 + ms; t += 33) {
    const r = s.update(typeof result === 'function' ? result(t) : result, t);
    events.push(...r.events);
    state = r.state;
  }
  return { state, events, end: t0 + ms };
}

describe('PoseStabilizer', () => {
  it('waits for the enter window before showing a pose', () => {
    const s = new PoseStabilizer();
    expect(s.update(pose('peace'), 0).state.kind).not.toBe('pose');
    expect(s.update(pose('peace'), 33).state.kind).not.toBe('pose');
    expect(s.update(pose('peace'), 100).state.kind).not.toBe('pose');
    const r = s.update(pose('peace'), 170);
    expect(r.state).toMatchObject({ kind: 'pose', pose: 'peace' });
    expect(r.events).toEqual([{ type: 'enter', pose: 'peace' }]);
  });

  it('does not flicker when frames alternate between two poses', () => {
    const s = new PoseStabilizer();
    const { events } = feed(s, (t) => (Math.round(t / 33) % 2 ? pose('peace') : pose('three')), 0, 2000);
    expect(events.filter((e) => e.type === 'enter')).toEqual([]);
  });

  it('ignores low-confidence frames', () => {
    const s = new PoseStabilizer();
    const { state } = feed(s, pose('peace', 0.4), 0, 1000);
    expect(state.kind).not.toBe('pose');
  });

  it('survives brief drop-outs (hysteresis) but releases after the release window', () => {
    const s = new PoseStabilizer();
    let r = feed(s, pose('point'), 0, 300);
    expect(r.state).toMatchObject({ kind: 'pose', pose: 'point' });
    r = feed(s, unknown, r.end + 33, 200); // shorter than releaseMs
    expect(r.state).toMatchObject({ kind: 'pose', pose: 'point' });
    expect(r.events).toEqual([]);
    r = feed(s, unknown, r.end + 33, 300);
    expect(r.state).toEqual({ kind: 'unknown', reason: 'unrecognized' });
    expect(r.events).toEqual([{ type: 'exit', pose: 'point' }]);
  });

  it('switches directly between poses with exit + enter events', () => {
    const s = new PoseStabilizer();
    let r = feed(s, pose('peace'), 0, 300);
    r = feed(s, pose('rockon'), r.end + 33, 300);
    expect(r.state).toMatchObject({ kind: 'pose', pose: 'rockon' });
    expect(r.events).toEqual([
      { type: 'exit', pose: 'peace' },
      { type: 'enter', pose: 'rockon' },
    ]);
  });

  it('requires experimental poses to be held longer', () => {
    const s = new PoseStabilizer();
    let r = feed(s, pose('rodrick'), 0, 250);
    expect(r.state.kind).not.toBe('pose');
    r = feed(s, pose('rodrick'), r.end + 33, 150);
    expect(r.state).toMatchObject({ kind: 'pose', pose: 'rodrick' });
  });

  it('distinguishes "no hands" from "hands without a pose" and goes idle after the idle delay', () => {
    const s = new PoseStabilizer();
    expect(feed(s, unknown, 0, 100).state.kind).toBe('unknown');
    let r = feed(s, none, 200, 1000);
    expect(r.state.kind).toBe('none');
    r = feed(s, none, r.end + 33, 800);
    expect(r.state.kind).toBe('idle');
    expect(r.events).toEqual([{ type: 'enter', pose: 'idle' }]);
    const back = s.update(unknown, r.end + 33);
    expect(back.events).toEqual([{ type: 'exit', pose: 'idle' }]);
    expect(back.state.kind).toBe('unknown');
  });

  it('emits a single held event once a pose is held long enough', () => {
    const s = new PoseStabilizer();
    const { events } = feed(s, pose('hello'), 0, 3000);
    expect(events.filter((e) => e.type === 'held')).toEqual([{ type: 'held', pose: 'hello' }]);
  });

  it('reset clears the active pose', () => {
    const s = new PoseStabilizer();
    feed(s, pose('hello'), 0, 400);
    s.reset(500);
    expect(s.state().kind).toBe('none');
  });
});

describe('Cooldown', () => {
  it('blocks repeats within the window and allows them after', () => {
    const c = new Cooldown(1000);
    expect(c.take('peace', 0)).toBe(true);
    expect(c.take('peace', 500)).toBe(false);
    expect(c.take('rockon', 500)).toBe(true);
    expect(c.take('peace', 1001)).toBe(true);
  });
});
