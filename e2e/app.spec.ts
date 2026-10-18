import { expect, test, type Page } from '@playwright/test';

/** Counts getUserMedia calls and keeps every stream so tests can check tracks were stopped. */
async function instrumentCamera(page: Page, mode: 'real' | 'deny' | 'unsupported' = 'real') {
  await page.addInitScript((m) => {
    const w = window as unknown as { __gum: number; __streams: MediaStream[] };
    w.__gum = 0;
    w.__streams = [];
    if (m === 'unsupported') {
      Object.defineProperty(navigator, 'mediaDevices', { value: undefined, configurable: true });
      return;
    }
    const md = navigator.mediaDevices;
    const orig = md.getUserMedia.bind(md);
    md.getUserMedia = async (c?: MediaStreamConstraints) => {
      w.__gum++;
      if (m === 'deny') throw new DOMException('Permission denied', 'NotAllowedError');
      const s = await orig(c);
      w.__streams.push(s);
      return s;
    };
  }, mode);
}

const gumCalls = (page: Page) => page.evaluate(() => (window as unknown as { __gum: number }).__gum);
const liveTracks = (page: Page) =>
  page.evaluate(() =>
    (window as unknown as { __streams: MediaStream[] }).__streams.flatMap((s) => s.getTracks()).filter((t) => t.readyState === 'live').length,
  );

test('landing page explains the app and never asks for the camera on load', async ({ page }) => {
  await instrumentCamera(page);
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Your camera.');
  await expect(page.getByText('Your camera stays on your device.').first()).toBeVisible();
  await expect(page.getByRole('link', { name: 'Enter the camera' })).toBeVisible();
  await page.waitForTimeout(800);
  expect(await gumCalls(page)).toBe(0);
});

test('navigation, direct loads and unknown routes', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('link', { name: 'Explore poses' }).click();
  await expect(page).toHaveURL(/\/poses$/);
  await expect(page.getByRole('heading', { name: 'Every pose, and what it does' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Peace' })).toBeVisible();
  await page.getByRole('button', { name: 'Experimental' }).click();
  await expect(page.getByRole('heading', { name: 'Peace' })).toHaveCount(0);
  await expect(page.getByRole('heading', { name: 'Rodrick' })).toBeVisible();

  await page.reload();
  await expect(page.getByRole('heading', { name: 'Every pose, and what it does' })).toBeVisible();

  await page.goto('/help#privacy');
  await expect(page.getByRole('heading', { name: 'Privacy' })).toBeVisible();

  await page.goto('/studio');
  await expect(page.getByRole('button', { name: 'Turn on camera' })).toBeVisible();

  await page.goto('/definitely-not-here');
  await expect(page.getByRole('heading', { name: 'This page doesn’t exist' })).toBeVisible();
});

test('opening /studio directly waits for an explicit click before using the camera', async ({ page }) => {
  await instrumentCamera(page);
  await page.goto('/studio');
  await page.waitForTimeout(800);
  expect(await gumCalls(page)).toBe(0);
});

test('denied camera permission shows a clear, recoverable error', async ({ page }) => {
  await instrumentCamera(page, 'deny');
  await page.goto('/');
  await page.getByRole('link', { name: 'Enter the camera' }).click();
  await expect(page.getByRole('heading', { name: 'Camera permission was blocked' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Try again' })).toBeVisible();
  await expect(page.getByRole('link', { name: 'Camera help' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Take photo' })).toBeDisabled();
});

test('unsupported browsers get an explanation instead of a black box', async ({ page }) => {
  await instrumentCamera(page, 'unsupported');
  await page.goto('/studio');
  await page.getByRole('button', { name: 'Turn on camera' }).click();
  await expect(page.getByRole('heading', { name: 'This browser can’t open a camera' })).toBeVisible();
});

test('full pipeline: camera, local hand tracking, photo, and camera release', async ({ page }) => {
  test.setTimeout(150_000);
  await instrumentCamera(page);
  const external: string[] = [];
  page.on('request', (r) => {
    const u = r.url();
    if (!u.startsWith('http://localhost:4173') && !u.startsWith('blob:') && !u.startsWith('data:')) external.push(u);
  });
  await page.goto('/');
  await page.getByRole('link', { name: 'Enter the camera' }).click();
  await expect(page.getByRole('button', { name: 'Turn camera off' })).toBeVisible({ timeout: 15_000 });
  // the model is loaded from this origin; in headless CI the GPU is emulated, so allow time
  await expect(page.locator('.studio__status')).toContainText('Show a hand', { timeout: 120_000 });
  await page.keyboard.press('d');
  await expect(page.locator('.debug')).toContainText(/backend (worker|main)-(gpu|cpu)/, { timeout: 10_000 });
  // a completed inference on the fake camera's test pattern reports "no hands"
  await expect(page.locator('.debug')).toContainText('frame no hands', { timeout: 120_000 });
  test.info().annotations.push({ type: 'detector', description: await page.locator('.debug').innerText() });

  await page.getByRole('button', { name: 'Take photo' }).click();
  const photo = page.getByRole('img', { name: /Captured photo/ });
  await expect(photo).toBeVisible();
  const size = await photo.evaluate((img: HTMLImageElement) => [img.naturalWidth, img.naturalHeight]);
  expect(size[0]).toBeGreaterThan(300);
  await expect(page.getByRole('button', { name: 'Download PNG' })).toBeVisible();
  await page.getByRole('button', { name: 'Retake' }).click();
  await expect(photo).toHaveCount(0);

  expect(await liveTracks(page)).toBe(1);
  await page.getByRole('button', { name: 'Exit camera' }).click();
  await expect(page).toHaveURL(/\/$/);
  expect(await liveTracks(page)).toBe(0);
  expect(external).toEqual([]);
});

test('pose editor: create a custom pose, keep it after reload, export and import a pack', async ({ page }, testInfo) => {
  await page.goto('/admin');
  await expect(page.getByRole('heading', { name: 'Pose editor' })).toBeVisible();
  await page.getByRole('button', { name: 'New pose' }).click();
  await page.getByLabel('Name').fill('Pinky Promise');
  for (const f of ['Index', 'Middle', 'Ring']) {
    await page.getByRole('radiogroup', { name: f }).getByRole('radio', { name: 'Folded' }).click();
  }
  await page.getByLabel('New caption').fill('I promise.');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.locator('.ed-tape', { hasText: 'I promise.' })).toBeVisible();
  await expect(page.locator('.ed-save')).toHaveText('Saved in this browser');

  await page.reload();
  await expect(page.getByRole('button', { name: /Pinky Promise/ })).toBeVisible();

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export pack' }).click();
  const download = await downloadPromise;
  const file = testInfo.outputPath('pack.json');
  await download.saveAs(file);
  const json = JSON.parse(await (await import('node:fs/promises')).readFile(file, 'utf8'));
  expect(json.format).toBe('posecam-pack');
  expect(json.custom[0]).toMatchObject({ name: 'Pinky Promise', captions: ['NEW POSE UNLOCKED.', 'I promise.'] });

  // reset, then import the exported pack back
  await page.getByRole('button', { name: 'Reset all' }).click();
  await page.getByRole('button', { name: 'Reset everything' }).click();
  await expect(page.getByRole('button', { name: /Pinky Promise/ })).toHaveCount(0);
  await page.locator('input[type=file][accept*="json"]').setInputFiles(file);
  await page.getByRole('button', { name: 'Import pack' }).click();
  await expect(page.getByRole('button', { name: /Pinky Promise/ })).toBeVisible();

  // it shows up in the pose guide and the poses page
  await page.goto('/poses');
  await expect(page.getByRole('heading', { name: 'Pinky Promise' })).toBeVisible();
});
