import { Download, Link2, RefreshCcw, Share2, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { CapturedPhoto } from '../../engine/StudioEngine';
import { canvasToBlob, downloadBlob, photoFilename, shareLink, sharePhoto } from '../../render/capture';
import { pushToast } from '../../state/toasts';
import type { PhotoFormat } from '../../state/settings';

export function CaptureDialog({ photo, onClose }: { photo: CapturedPhoto; onClose: () => void }) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const [canShareFiles, setCanShareFiles] = useState(false);
  const filename = useRef(photoFilename(photo.format)).current;

  useEffect(() => {
    try {
      const f = new File([photo.blob], filename, { type: photo.blob.type });
      setCanShareFiles(typeof navigator.canShare === 'function' && navigator.canShare({ files: [f] }));
    } catch {
      setCanShareFiles(false);
    }
  }, [photo, filename]);

  // focus management: focus the dialog, trap Tab, close on Escape, restore focus
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    const el = dialogRef.current;
    el?.querySelector<HTMLElement>('[data-autofocus]')?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
      }
      if (e.key === 'Tab' && el) {
        const items = [...el.querySelectorAll<HTMLElement>('button, a[href]')].filter((x) => !x.hasAttribute('disabled'));
        if (!items.length) return;
        const first = items[0];
        const last = items[items.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('keydown', onKey);
      prev?.focus?.();
    };
  }, [onClose]);

  const download = async (format: PhotoFormat) => {
    try {
      if (format === photo.format) {
        downloadBlob(photo.blob, filename);
      } else {
        const img = new Image();
        img.src = photo.url;
        await img.decode();
        const c = document.createElement('canvas');
        c.width = photo.width;
        c.height = photo.height;
        c.getContext('2d')!.drawImage(img, 0, 0);
        downloadBlob(await canvasToBlob(c, format), photoFilename(format));
      }
      pushToast({ title: 'Photo downloaded' });
    } catch {
      pushToast({ title: 'Download failed', body: 'Try again, or long-press the photo to save it.' });
    }
  };

  const appUrl = window.location.origin;

  return (
    <div className="dialog-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div ref={dialogRef} className="dialog" role="dialog" aria-modal="true" aria-labelledby="capture-title">
        <div className="dialog__head">
          <h2 id="capture-title">Your photo</h2>
          <button type="button" className="icon-btn" aria-label="Close photo" onClick={onClose}>
            <X />
          </button>
        </div>
        <img className="dialog__photo" src={photo.url} alt="Captured photo with your selected stickers and effects" width={photo.width} height={photo.height} />
        <div className="dialog__actions">
          <button type="button" className="btn btn--primary" data-autofocus onClick={() => void download('png')}>
            <Download aria-hidden="true" />
            Download PNG
          </button>
          <button type="button" className="btn" onClick={() => void download('jpeg')}>
            <Download aria-hidden="true" />
            Download JPEG
          </button>
          {canShareFiles ? (
            <button
              type="button"
              className="btn"
              onClick={async () => {
                const r = await sharePhoto(photo.blob, filename, appUrl);
                if (r === 'unavailable') pushToast({ title: 'Sharing isn’t available here', body: 'Download the photo instead.' });
              }}
            >
              <Share2 aria-hidden="true" />
              Share photo
            </button>
          ) : null}
          <button
            type="button"
            className="btn"
            onClick={async () => {
              const r = await shareLink(appUrl);
              if (r === 'copied') pushToast({ title: 'Link copied', body: appUrl });
              else if (r === 'unavailable') pushToast({ title: 'Couldn’t copy the link', body: appUrl });
            }}
          >
            <Link2 aria-hidden="true" />
            Share POSE CAM
          </button>
          <span className="spacer" />
          <button type="button" className="btn btn--ghost" onClick={onClose}>
            <RefreshCcw aria-hidden="true" />
            Retake
          </button>
        </div>
        <p className="dialog__note">This photo exists only in your browser until you download or share it.</p>
      </div>
    </div>
  );
}
