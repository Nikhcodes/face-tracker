/** Minimal History-API router (four routes don't justify a routing dependency). */
import { useSyncExternalStore, type AnchorHTMLAttributes, type MouseEvent } from 'react';

const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
if (typeof window !== 'undefined') window.addEventListener('popstate', notify);

export function navigate(to: string, state: Record<string, unknown> | null = null, replace = false): void {
  if (replace) window.history.replaceState(state, '', to);
  else window.history.pushState(state, '', to);
  notify();
  if (!to.includes('#')) window.scrollTo(0, 0);
}

export function usePath(): string {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => window.location.pathname,
    () => '/',
  );
}

type LinkProps = AnchorHTMLAttributes<HTMLAnchorElement> & { to: string; state?: Record<string, unknown> };

export function Link({ to, state, onClick, ...rest }: LinkProps) {
  const path = usePath();
  const handle = (e: MouseEvent<HTMLAnchorElement>) => {
    onClick?.(e);
    if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (to.startsWith('#')) return;
    e.preventDefault();
    navigate(to, state ?? null);
  };
  return <a href={to} aria-current={path === to ? 'page' : undefined} onClick={handle} {...rest} />;
}
