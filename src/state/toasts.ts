import { createStore } from '../lib/store';

export interface ToastItem {
  id: number;
  title: string;
  body?: string;
  tone?: 'default' | 'accent' | 'gold';
}

export const toastStore = createStore<ToastItem[]>([]);
let nextId = 1;

export function pushToast(t: Omit<ToastItem, 'id'> & { id?: number }, ms = 3000): void {
  const id = nextId++;
  toastStore.set((list) => [...list.slice(-1), { ...t, id }]);
  setTimeout(() => toastStore.set((list) => list.filter((x) => x.id !== id)), ms);
}
