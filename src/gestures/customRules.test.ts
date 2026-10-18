import { describe, expect, it } from 'vitest';
import { classifyFrame } from './classify';
import { DEFAULT_RULE, matchRule, ruleFromFeatures, ruleIssues, type CustomRule } from './customRules';
import { computeFeatures } from './features';
import { SPECS, rollToPointThumb, synthHand } from './synthetic';

const feat = (spec: keyof typeof SPECS, opts = {}) => computeFeatures(synthHand(SPECS[spec], { cy: 0.35, ...opts }), 16 / 9);
const pinkyOnly: CustomRule = { ...DEFAULT_RULE, thumb: 'down', index: 'down', middle: 'down', ring: 'down', pinky: 'up' };

describe('custom pose rules', () => {
  it('flags rules that would match almost anything', () => {
    const loose: CustomRule = { ...DEFAULT_RULE, thumb: 'any', index: 'up', middle: 'any', ring: 'any', pinky: 'any' };
    expect(ruleIssues(loose)).toHaveLength(1);
    expect(ruleIssues({ ...DEFAULT_RULE, thumb: 'down', thumbDirection: 'up' })).toHaveLength(1);
    expect(ruleIssues(pinkyOnly)).toEqual([]);
  });

  it('matches its finger pattern in any orientation, and nothing else', () => {
    // a "pinky promise" pose: only the pinky up
    const pinky = { index: SPECS.fist.index, middle: SPECS.fist.middle, ring: SPECS.fist.ring, pinky: SPECS.rockon.pinky, thumb: SPECS.fist.thumb };
    for (const roll of [0, 45, 90, 180]) {
      expect(matchRule(pinkyOnly, [computeFeatures(synthHand(pinky, { roll }), 16 / 9)])).not.toBeNull();
    }
    for (const other of ['open', 'peace', 'point', 'rockon', 'fist'] as const) {
      expect(matchRule(pinkyOnly, [feat(other)])).toBeNull();
    }
  });

  it('respects thumb direction and frame zone', () => {
    const rule: CustomRule = { ...DEFAULT_RULE, thumb: 'up', index: 'down', middle: 'down', ring: 'down', pinky: 'down', thumbDirection: 'down' };
    const down = rollToPointThumb(SPECS.thumb, 'down');
    const up = rollToPointThumb(SPECS.thumb, 'up');
    expect(matchRule(rule, [feat('thumb', { roll: down })])).not.toBeNull();
    expect(matchRule(rule, [feat('thumb', { roll: up })])).toBeNull();
    const upper: CustomRule = { ...DEFAULT_RULE, zone: 'upper', thumb: 'any' };
    expect(matchRule(upper, [feat('open', { cy: 0.3 })])).not.toBeNull();
    expect(matchRule(upper, [feat('open', { cy: 0.75 })])).toBeNull();
  });

  it('two-hand rules need both hands', () => {
    const rule: CustomRule = { ...pinkyOnly, hands: 2, pinky: 'down', index: 'up' };
    const a = feat('point', { cx: 0.3 });
    const b = feat('point', { cx: 0.7, handedness: 'Left' });
    expect(matchRule(rule, [a])).toBeNull();
    expect(matchRule(rule, [a, b])).not.toBeNull();
  });

  it('teach-from-camera turns a live hand into a rule that matches it', () => {
    const f = feat('rockon');
    const rule = ruleFromFeatures(f);
    expect(rule).toMatchObject({ index: 'up', middle: 'down', ring: 'down', pinky: 'up' });
    expect(matchRule(rule, [f])).not.toBeNull();
  });

  it('classifyFrame checks custom poses first and skips disabled built-ins', () => {
    const hand = synthHand(SPECS.peace, { cy: 0.35 });
    const peaceLike: CustomRule = { ...DEFAULT_RULE, thumb: 'any', index: 'up', middle: 'up', ring: 'down', pinky: 'down' };
    const custom = [{ id: 'custom-v' as const, rule: peaceLike }];
    const withCustom = classifyFrame({ hands: [hand], aspect: 16 / 9 }, { custom });
    expect(withCustom).toMatchObject({ kind: 'pose', pose: 'custom-v' });
    const disabled = classifyFrame({ hands: [hand], aspect: 16 / 9 }, { disabled: new Set(['peace']) });
    expect(disabled.kind).toBe('unknown');
  });
});
