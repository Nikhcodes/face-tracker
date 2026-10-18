# Custom poses and pose packs

The pose editor (`/admin`) lets anyone:

- create new poses from a finger pattern;
- add their own pictures to any pose;
- rewrite captions;
- choose effects and sounds;
- switch built-in poses off.

## Where changes live

| Layer | Stored in | Who sees it |
| --- | --- | --- |
| Local edits | the browser's IndexedDB | only that browser |
| Site pack | `public/packs/default.json` in the repo | every visitor |
| Built-ins | `src/config/poses.ts` | every visitor |

If a browser has local edits, they replace the site pack for that browser. **Reset all** in the editor drops them and brings the site pack back.

Anyone can open `/admin` on your deployed site. Their edits only change their own browser, never your site.

## Publishing for everyone

1. Make your changes in the editor (locally with `npm run dev`, or on the live site).
2. Click **Export pack**. You get `posecam-pack-YYYY-MM-DD.json`, with pictures embedded.
3. Save it as `public/packs/default.json`.
4. Commit and push. Vercel redeploys.

To make packs smaller, you can put image files in `public/overlays/` and reference them by path (`"src": "/overlays/my-sticker.webp"`) instead of embedding `data:` URLs.

## Custom pose rules

A custom pose is a finger pattern, matched with the same orientation-independent features as the built-ins:

| Field | Values |
| --- | --- |
| `hands` | `1` (either hand) or `2` (both must match) |
| `thumb`, `index`, `middle`, `ring`, `pinky` | `up`, `down` (folded) or `any` |
| `thumbDirection` | `any`, `up`, `down` (needs `thumb: up`) |
| `zone` | `any`, `upper`, `lower` (half of the frame where the hand centre must be) |

At least two digits must be `up` or `down`, otherwise the rule would match almost any hand. Custom poses are checked **before** built-ins, so a custom pose with the same shape as Peace replaces Peace while it's on. The editor warns about this.

**Copy my hand shape** in the camera tester fills the rule from your live hand. Digits the tracker is unsure about become `any`.

## Pack format (version 1)

```jsonc
{
  "format": "posecam-pack",
  "version": 1,
  "name": "My poses",
  "updatedAt": "2026-10-18T20:00:00.000Z",
  "overrides": {
    "peace": {
      "name": "V Sign",
      "captions": ["V FOR VICTORY."],
      "stickers": [{ "src": "data:image/webp;base64,…", "width": 720, "height": 540 }],
      "keepOriginal": false,        // hide the built-in pictures
      "particles": "stars",         // none | sparkles | stars | confetti | hearts | notes | rain
      "shake": true,
      "spotlight": false,
      "sound": "chime",             // pop | chime | boom | pew | sad | sparkle | none
      "disabled": false
    },
    "rodrick": { "disabled": true }
  },
  "custom": [
    {
      "id": "custom-mz3k2-a8f1q",
      "name": "Pinky Promise",
      "blurb": "A promise is a promise.",
      "howTo": "",
      "rule": { "hands": 1, "thumb": "down", "index": "down", "middle": "down", "ring": "down", "pinky": "up", "thumbDirection": "any", "zone": "any" },
      "captions": ["I PROMISE."],
      "stickers": [],
      "sound": "chime",
      "particles": "hearts",
      "shake": false,
      "spotlight": false,
      "disabled": false
    }
  ]
}
```

Built-in ids you can override: `hello`, `peace`, `point`, `thumbsup`, `thumbsdown`, `rockon`, `three`, `shaka`, `fingergun`, `think`, `doublethumbs`, `celebrate`, `rodrick`, `heart`, `idle`.

## Validation

Every pack, whether imported, loaded from the site or read from storage, goes through `sanitizePack()` (`src/config/pack.ts`):

- **Images:** only `data:image/(png|jpeg|webp|gif);base64` URLs (up to about 2 MB each) or site-relative image paths. No external URLs, so the canvas can't be tainted and nothing is fetched from elsewhere.
- **Text:** trimmed, control characters removed, and capped. Names 40 characters, captions 48 characters (8 per pose).
- **Counts:** at most 6 pictures per pose and 40 custom poses.
- **Rules:** unknown pose ids, invalid rules and duplicate ids are dropped, with warnings shown on import.
