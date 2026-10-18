import { describe, expect, it } from 'vitest';
import { emptyPack } from '../config/pack';
import { addCustomPose, applyPatch, deleteCustomPose, draftFor, duplicateCustomPose, packImageBytes, resetBuiltin } from './editPack';

const IMG = { src: 'data:image/png;base64,AAAA', width: 10, height: 10 };

describe('editPack', () => {
  it('shows built-in defaults until something changes', () => {
    const d = draftFor(emptyPack(), 'peace')!;
    expect(d).toMatchObject({ kind: 'builtin', name: 'Peace', enabled: true, edited: false, keepOriginal: true, particles: 'sparkles' });
  });

  it('stores only real changes for built-ins, and reverting removes the override', () => {
    let pack = applyPatch(emptyPack(), 'peace', { name: 'V Sign', captions: ['V.'] });
    expect(pack.overrides.peace).toEqual({ name: 'V Sign', captions: ['V.'] });
    pack = applyPatch(pack, 'peace', { name: 'Peace', captions: ['PLEASE REMAIN ICONIC.', 'PEACE AND LOVE, ONLY.'] });
    expect(pack.overrides.peace).toBeUndefined();
  });

  it('can switch a built-in off and back on', () => {
    let pack = applyPatch(emptyPack(), 'rodrick', { enabled: false });
    expect(pack.overrides.rodrick).toEqual({ disabled: true });
    expect(draftFor(pack, 'rodrick')!.enabled).toBe(false);
    pack = applyPatch(pack, 'rodrick', { enabled: true });
    expect(pack.overrides.rodrick).toBeUndefined();
  });

  it('adds pictures and can drop the originals', () => {
    const pack = applyPatch(emptyPack(), 'hello', { stickers: [IMG], keepOriginal: false });
    expect(pack.overrides.hello).toEqual({ stickers: [IMG], keepOriginal: false });
    expect(packImageBytes(pack)).toBe(Math.round(IMG.src.length * 0.75));
    expect(resetBuiltin(pack, 'hello').overrides.hello).toBeUndefined();
  });

  it('creates, edits, duplicates and deletes custom poses', () => {
    const { pack, id } = addCustomPose(emptyPack());
    expect(draftFor(pack, id)).toMatchObject({ kind: 'custom', name: 'My pose', enabled: true });
    const edited = applyPatch(pack, id, { name: 'Pinky', enabled: false, captions: ['P.'] });
    expect(edited.custom[0]).toMatchObject({ name: 'Pinky', disabled: true, captions: ['P.'] });
    const dup = duplicateCustomPose(edited, id)!;
    expect(dup.pack.custom).toHaveLength(2);
    expect(dup.pack.custom[1]).toMatchObject({ name: 'Pinky copy' });
    expect(dup.id).not.toBe(id);
    expect(deleteCustomPose(dup.pack, id).custom.map((c) => c.id)).toEqual([dup.id]);
  });
});
