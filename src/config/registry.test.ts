import { describe, expect, it } from 'vitest';
import { DEFAULT_RULE, type CustomRule } from '../gestures/customRules';
import { emptyPack, sanitizePack, type CustomPoseData, type PosePack } from './pack';
import { buildRegistry, describeRule, findConflicts, packToJson } from './registry';

const PNG = 'data:image/png;base64,iVBORw0KGgo=';
const custom = (over: Partial<CustomPoseData> = {}): CustomPoseData => ({
  id: 'custom-abc',
  name: 'Pinky Promise',
  blurb: '',
  howTo: '',
  rule: { ...DEFAULT_RULE, thumb: 'down', index: 'down', middle: 'down', ring: 'down', pinky: 'up' },
  captions: ['I PROMISE.'],
  stickers: [],
  sound: 'chime',
  particles: 'hearts',
  shake: false,
  spotlight: false,
  disabled: false,
  ...over,
});

describe('sanitizePack', () => {
  it('rejects files that are not packs', () => {
    expect(sanitizePack(null).pack).toBeNull();
    expect(sanitizePack({ format: 'other' }).pack).toBeNull();
    expect(sanitizePack({ format: 'posecam-pack', version: 99 }).pack).toBeNull();
  });

  it('keeps valid data and drops anything unsafe or unknown', () => {
    const { pack, warnings } = sanitizePack({
      format: 'posecam-pack',
      version: 1,
      name: 'Test',
      overrides: {
        peace: {
          name: 'V Sign',
          captions: ['ONE', 42, 'x'.repeat(200)],
          stickers: [
            { src: PNG, width: 100, height: 80 },
            { src: 'javascript:alert(1)', width: 10, height: 10 },
            { src: 'https://evil.example/x.png', width: 10, height: 10 },
            { src: '/overlays/../../etc.png', width: 10, height: 10 },
            { src: '/overlays/mine.webp', width: 300, height: 300 },
          ],
          particles: 'lasers',
          disabled: true,
        },
        notARealPose: { name: 'nope' },
      },
      custom: [custom(), custom(), { ...custom({ id: 'custom-bad' }), rule: { ...DEFAULT_RULE, index: 'any', middle: 'any', ring: 'any', pinky: 'any', thumb: 'up' } }, { id: 'evil', name: 'x' }],
    });
    expect(pack).not.toBeNull();
    const peace = pack!.overrides.peace!;
    expect(peace.name).toBe('V Sign');
    expect(peace.captions).toEqual(['ONE', 'x'.repeat(48)]);
    expect(peace.stickers!.map((s) => s.src)).toEqual([PNG, '/overlays/mine.webp']);
    expect(peace.particles).toBeUndefined();
    expect(peace.disabled).toBe(true);
    expect(Object.keys(pack!.overrides)).toEqual(['peace']);
    expect(pack!.custom.map((c) => c.id)).toEqual(['custom-abc']);
    expect(warnings.length).toBeGreaterThan(0);
  });

  it('round-trips through JSON export', () => {
    const pack: PosePack = { ...emptyPack('Mine'), custom: [custom()], overrides: { rockon: { captions: ['LOUD.'] } } };
    const again = sanitizePack(JSON.parse(packToJson(pack))).pack!;
    expect(again.custom).toEqual(pack.custom);
    expect(again.overrides).toEqual(pack.overrides);
  });
});

describe('buildRegistry', () => {
  it('applies overrides, adds custom poses and tracks disabled ones', () => {
    const pack: PosePack = {
      ...emptyPack(),
      overrides: {
        peace: { name: 'V Sign', stickers: [{ src: PNG, width: 10, height: 10 }], keepOriginal: false, particles: 'none', shake: true },
        point: { stickers: [{ src: PNG, width: 10, height: 10 }] },
        rodrick: { disabled: true },
      },
      custom: [custom(), custom({ id: 'custom-off', disabled: true })],
    };
    const reg = buildRegistry(pack);
    const peace = reg.byId.get('peace')!;
    expect(peace.name).toBe('V Sign');
    expect(peace.stickers).toEqual([{ kind: 'image', src: PNG, width: 10, height: 10 }]);
    expect(reg.byId.get('point')!.stickers).toHaveLength(3); // 2 originals + 1 added
    expect(peace.reaction.particles).toBeUndefined();
    expect(peace.reaction.shake).toBeGreaterThan(0);
    expect(reg.poses.find((p) => p.id === 'rodrick')).toBeUndefined();
    expect(reg.all.find((p) => p.id === 'rodrick')).toBeDefined();
    expect(reg.custom.map((c) => c.id)).toEqual(['custom-abc']);
    const mine = reg.byId.get('custom-abc')!;
    expect(mine).toMatchObject({ status: 'custom', name: 'Pinky Promise', hands: 1 });
    expect(mine.stickers[0]).toMatchObject({ kind: 'stamp', label: 'PINKY' });
    expect(mine.reaction.particles?.kind).toBe('hearts');
    expect(reg.poses[reg.poses.length - 1].id).toBe('idle');
  });

  it('warns when a custom rule would also catch a built-in pose', () => {
    const reg = buildRegistry(emptyPack());
    const peaceLike: CustomRule = { ...DEFAULT_RULE, thumb: 'any', index: 'up', middle: 'up', ring: 'down', pinky: 'down' };
    expect(findConflicts(peaceLike, reg).map((p) => p.id)).toContain('peace');
    const pinky: CustomRule = custom().rule;
    expect(findConflicts(pinky, reg)).toEqual([]);
  });

  it('describes rules in plain words', () => {
    expect(describeRule(custom().rule)).toBe('Thumb folded, index folded, middle folded, ring folded, pinky up.');
  });
});
