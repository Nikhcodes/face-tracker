import { useStore } from '../lib/useStore';
import { toastStore } from '../state/toasts';

export function Toasts() {
  const toasts = useStore(toastStore);
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className="toast" data-tone={t.tone ?? 'default'}>
          <span className="toast__title">{t.title}</span>
          {t.body ? <span className="toast__body">{t.body}</span> : null}
        </div>
      ))}
    </div>
  );
}
