import { Plus, X } from 'lucide-react';
import { useState } from 'react';
import { LIMITS } from '../../config/pack';

export function CaptionEditor({ captions, onChange }: { captions: string[]; onChange: (c: string[]) => void }) {
  const [text, setText] = useState('');
  const add = () => {
    const t = text.trim().slice(0, LIMITS.caption);
    if (!t || captions.length >= LIMITS.captions) return;
    onChange([...captions, t]);
    setText('');
  };
  return (
    <div className="ed-captions">
      {captions.length ? (
        <ul className="ed-tapes" aria-label="Captions">
          {captions.map((c, i) => (
            <li key={`${i}-${c}`} className="ed-tape">
              <span>{c}</span>
              <button type="button" aria-label={`Remove caption ${c}`} onClick={() => onChange(captions.filter((_, j) => j !== i))}>
                <X />
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="ed-help">No captions: this pose reacts without text.</p>
      )}
      {captions.length < LIMITS.captions ? (
        <form
          className="ed-inline"
          onSubmit={(e) => {
            e.preventDefault();
            add();
          }}
        >
          <label className="sr-only" htmlFor="caption-input">
            New caption
          </label>
          <input
            id="caption-input"
            className="ed-input"
            value={text}
            maxLength={LIMITS.caption}
            placeholder="ABSOLUTE CINEMA."
            onChange={(e) => setText(e.target.value)}
          />
          <button type="submit" className="btn btn--sm" disabled={!text.trim()}>
            <Plus aria-hidden="true" />
            Add
          </button>
        </form>
      ) : null}
      <p className="ed-help">One is picked at random each time. Short and loud works best; they’re shown in capitals.</p>
    </div>
  );
}
