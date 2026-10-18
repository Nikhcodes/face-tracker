import { ImagePlus, X } from 'lucide-react';
import { useRef, useState, type DragEvent } from 'react';
import { LIMITS, type StickerImage } from '../../config/pack';
import type { StickerRef } from '../../config/poses';
import { ACCEPTED_IMAGES, fileToSticker } from '../../lib/image';
import { pushToast } from '../../state/toasts';
import { ToggleRow } from '../studio/Controls';

function Thumb({ src, label, onRemove }: { src: string; label: string; onRemove?: () => void }) {
  return (
    <li className="ed-thumb">
      <img src={src} alt={label} />
      {onRemove ? (
        <button type="button" className="ed-thumb__remove" aria-label={`Remove ${label}`} onClick={onRemove}>
          <X />
        </button>
      ) : null}
    </li>
  );
}

function originalSrc(s: StickerRef): string | null {
  return s.kind === 'stamp' ? null : s.src;
}

export function PictureEditor({
  kind,
  originals,
  keepOriginal,
  stickers,
  onChange,
}: {
  kind: 'builtin' | 'custom';
  originals: StickerRef[];
  keepOriginal: boolean;
  stickers: StickerImage[];
  onChange: (patch: { stickers?: StickerImage[]; keepOriginal?: boolean }) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const room = LIMITS.stickers - stickers.length;
  const originalImages = originals.map(originalSrc).filter((s): s is string => !!s);

  const add = async (files: FileList | File[]) => {
    const list = [...files].slice(0, Math.max(0, room));
    if (!list.length) {
      pushToast({ title: `Up to ${LIMITS.stickers} pictures per pose` });
      return;
    }
    setBusy(true);
    const added: StickerImage[] = [];
    for (const f of list) {
      try {
        added.push(await fileToSticker(f));
      } catch (err) {
        pushToast({ title: 'Couldn’t add that picture', body: err instanceof Error ? err.message : undefined });
      }
    }
    setBusy(false);
    if (added.length) onChange({ stickers: [...stickers, ...added] });
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDrag(false);
    if (e.dataTransfer.files.length) void add(e.dataTransfer.files);
  };

  return (
    <div className="ed-pictures">
      {kind === 'builtin' && originals.length ? (
        <div className="ed-pictures__group">
          <ToggleRow
            label="Use the original pictures"
            hint={originalImages.length ? undefined : 'This pose uses a drawn badge'}
            checked={keepOriginal}
            onChange={(v) => onChange({ keepOriginal: v })}
          />
          {keepOriginal && originalImages.length ? (
            <ul className="ed-thumbs" aria-label="Original pictures">
              {originalImages.map((src, i) => (
                <Thumb key={src} src={src} label={`Original picture ${i + 1}`} />
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      <div className="ed-pictures__group">
        {stickers.length ? (
          <ul className="ed-thumbs" aria-label="Your pictures">
            {stickers.map((s, i) => (
              <Thumb key={`${i}-${s.src.slice(-24)}`} src={s.src} label={`Your picture ${i + 1}`} onRemove={() => onChange({ stickers: stickers.filter((_, j) => j !== i) })} />
            ))}
          </ul>
        ) : null}
        {room > 0 ? (
          <div
            className="ed-drop"
            data-drag={drag}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={onDrop}
          >
            <ImagePlus aria-hidden="true" />
            <span>
              <strong>Drop pictures here</strong> or{' '}
              <button type="button" className="ed-linkbtn" onClick={() => input.current?.click()} disabled={busy}>
                {busy ? 'adding…' : 'choose files'}
              </button>
            </span>
            <span className="ed-drop__hint">PNG, JPEG, WebP or GIF. Resized to 720 px. GIFs become a still image.</span>
            <input
              ref={input}
              type="file"
              accept={ACCEPTED_IMAGES.join(',')}
              multiple
              className="sr-only"
              tabIndex={-1}
              onChange={(e) => {
                if (e.target.files) void add(e.target.files);
                e.target.value = '';
              }}
            />
          </div>
        ) : null}
        <p className="ed-help">
          {kind === 'custom' && !stickers.length
            ? 'No pictures yet, so this pose shows a badge with its name.'
            : 'With several pictures, one is picked at random each time.'}
        </p>
      </div>
    </div>
  );
}
