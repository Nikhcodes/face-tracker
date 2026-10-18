import { describe, expect, it, vi } from 'vitest';

describe('initPacks', () => {
  it('loads a valid site pack, ignores HTML fallbacks, and survives missing IndexedDB', async () => {
    vi.resetModules();
    const { initPacks, packState, registryStore } = await import('./registry');
    const site = {
      format: 'posecam-pack',
      version: 1,
      name: 'Site',
      overrides: { peace: { name: 'Site Peace' } },
      custom: [],
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(site), { headers: { 'content-type': 'application/json' } }));
    await initPacks(fetchMock as unknown as typeof fetch);
    expect(fetchMock).toHaveBeenCalledWith('/packs/default.json', { cache: 'no-cache' });
    expect(packState.get()).toMatchObject({ ready: true, storage: 'memory', local: null });
    expect(registryStore.get().byId.get('peace')?.name).toBe('Site Peace');
  });

  it('treats an SPA HTML response as "no site pack"', async () => {
    vi.resetModules();
    const { initPacks, packState } = await import('./registry');
    const fetchMock = vi.fn().mockResolvedValue(new Response('<!doctype html>', { headers: { 'content-type': 'text/html' } }));
    await initPacks(fetchMock as unknown as typeof fetch);
    expect(packState.get().site).toBeNull();
  });
});
