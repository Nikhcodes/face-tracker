import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS, resetEffects, sanitizeSettings } from './settings';

describe('settings', () => {
  it('falls back to defaults for garbage input', () => {
    expect(sanitizeSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(sanitizeSettings('nope')).toEqual(DEFAULT_SETTINGS);
    expect(sanitizeSettings({ filter: 'sepia', quality: 7, volume: 'loud' })).toEqual(DEFAULT_SETTINGS);
  });

  it('keeps valid values and clamps volume', () => {
    const s = sanitizeSettings({ filter: 'mono', grain: true, quality: 'high', volume: 4 });
    expect(s.filter).toBe('mono');
    expect(s.grain).toBe(true);
    expect(s.quality).toBe('high');
    expect(s.volume).toBe(1);
  });

  it('never restores audio as on (no autoplay after reload)', () => {
    const s = sanitizeSettings({ sound: true, music: true });
    expect(s.sound).toBe(false);
    expect(s.music).toBe(false);
  });

  it('reset effects restores visual settings but keeps camera/audio choices', () => {
    const custom = { ...DEFAULT_SETTINGS, filter: 'warm' as const, grain: true, landmarks: true, mirror: false, volume: 0.2 };
    const r = resetEffects(custom);
    expect(r.filter).toBe('original');
    expect(r.grain).toBe(false);
    expect(r.landmarks).toBe(false);
    expect(r.mirror).toBe(false);
    expect(r.volume).toBe(0.2);
  });
});
