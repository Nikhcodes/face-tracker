/**
 * Synthesised sound effects and an optional generative music loop.
 * Everything is generated with the Web Audio API — no audio files, nothing to
 * license, and nothing plays until the user turns sound on.
 */
export type Cue = 'pop' | 'chime' | 'boom' | 'pew' | 'sad' | 'sparkle' | 'shutter' | 'tick' | 'tickFinal' | 'unlock' | 'ui';

type Ctor = typeof AudioContext;

export class SoundEngine {
  private ctx: AudioContext | null = null;
  private master: GainNode | null = null;
  private sfx: GainNode | null = null;
  private musicBus: GainNode | null = null;
  private noise: AudioBuffer | null = null;
  private musicTimer: ReturnType<typeof setInterval> | null = null;
  private nextBeat = 0;
  private beat = 0;
  private volume = 0.6;
  enabled = false;

  static supported(): boolean {
    return typeof window !== 'undefined' && !!(window.AudioContext || (window as unknown as { webkitAudioContext?: Ctor }).webkitAudioContext);
  }

  /** Must be called from a user gesture (click/tap). */
  async enable(): Promise<boolean> {
    if (!SoundEngine.supported()) return false;
    if (!this.ctx) {
      const C: Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: Ctor }).webkitAudioContext;
      this.ctx = new C();
      const comp = this.ctx.createDynamicsCompressor();
      comp.threshold.value = -16;
      comp.ratio.value = 4;
      this.master = this.ctx.createGain();
      this.master.gain.value = this.volume;
      this.sfx = this.ctx.createGain();
      this.sfx.gain.value = 0.8;
      this.musicBus = this.ctx.createGain();
      this.musicBus.gain.value = 0.32;
      this.sfx.connect(this.master);
      this.musicBus.connect(this.master);
      this.master.connect(comp).connect(this.ctx.destination);
      const len = this.ctx.sampleRate;
      this.noise = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const data = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === 'suspended') await this.ctx.resume().catch(() => undefined);
    this.enabled = true;
    return true;
  }

  disable(): void {
    this.enabled = false;
    this.stopMusic();
  }

  setVolume(v: number): void {
    this.volume = v;
    if (this.master && this.ctx) this.master.gain.setTargetAtTime(v, this.ctx.currentTime, 0.05);
  }

  async dispose(): Promise<void> {
    this.disable();
    const ctx = this.ctx;
    this.ctx = null;
    await ctx?.close().catch(() => undefined);
  }

  private tone(freq: number, start: number, dur: number, type: OscillatorType, gain: number, endFreq?: number, bus?: AudioNode) {
    const ctx = this.ctx;
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, start);
    if (endFreq) osc.frequency.exponentialRampToValueAtTime(Math.max(20, endFreq), start + dur);
    g.gain.setValueAtTime(0.0001, start);
    g.gain.exponentialRampToValueAtTime(gain, start + Math.min(0.012, dur / 4));
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    osc.connect(g).connect(bus ?? this.sfx!);
    osc.start(start);
    osc.stop(start + dur + 0.05);
  }

  private noiseBurst(start: number, dur: number, gain: number, filter: BiquadFilterType, freq: number, bus?: AudioNode) {
    const ctx = this.ctx;
    if (!ctx || !this.noise) return;
    const src = ctx.createBufferSource();
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    f.type = filter;
    f.frequency.value = freq;
    const g = ctx.createGain();
    g.gain.setValueAtTime(gain, start);
    g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
    src.connect(f).connect(g).connect(bus ?? this.sfx!);
    src.start(start, Math.random() * 0.5);
    src.stop(start + dur + 0.05);
  }

  play(cue: Cue | 'none'): void {
    if (!this.enabled || !this.ctx || cue === 'none') return;
    const t = this.ctx.currentTime + 0.01;
    switch (cue) {
      case 'pop':
        this.tone(520, t, 0.12, 'sine', 0.5, 980);
        break;
      case 'ui':
        this.tone(880, t, 0.05, 'sine', 0.18);
        break;
      case 'chime':
        this.tone(1318.5, t, 0.5, 'triangle', 0.32);
        this.tone(1975.5, t + 0.09, 0.6, 'triangle', 0.24);
        break;
      case 'sparkle':
        [1568, 2093, 2637, 3136].forEach((f, i) => this.tone(f, t + i * 0.05, 0.25, 'sine', 0.18));
        break;
      case 'boom':
        this.tone(140, t, 0.45, 'sine', 0.9, 38);
        this.noiseBurst(t, 0.25, 0.4, 'lowpass', 900);
        break;
      case 'pew':
        this.tone(1400, t, 0.18, 'square', 0.18, 180);
        break;
      case 'sad':
        this.tone(392, t, 0.32, 'triangle', 0.35, 370);
        this.tone(311, t + 0.34, 0.6, 'triangle', 0.35, 277);
        break;
      case 'shutter':
        this.noiseBurst(t, 0.06, 0.6, 'highpass', 2500);
        this.noiseBurst(t + 0.07, 0.05, 0.4, 'highpass', 3500);
        break;
      case 'tick':
        this.tone(1000, t, 0.08, 'sine', 0.3);
        break;
      case 'tickFinal':
        this.tone(1500, t, 0.14, 'sine', 0.35);
        break;
      case 'unlock':
        [523.25, 659.25, 783.99, 1046.5].forEach((f, i) => this.tone(f, t + i * 0.08, 0.3, 'triangle', 0.28));
        break;
    }
  }

  // ── Music: a soft four-chord loop, scheduled ahead of time ────────────────
  private static readonly BPM = 84;
  private static readonly CHORDS = [
    [220.0, 261.63, 329.63, 392.0], // Am7
    [174.61, 220.0, 261.63, 329.63], // Fmaj7
    [130.81, 164.81, 196.0, 246.94], // Cmaj7
    [196.0, 246.94, 293.66, 349.23], // G7
  ];

  get musicPlaying(): boolean {
    return this.musicTimer !== null;
  }

  startMusic(): void {
    if (!this.enabled || !this.ctx || this.musicTimer) return;
    this.nextBeat = this.ctx.currentTime + 0.1;
    this.beat = 0;
    this.musicBus!.gain.setTargetAtTime(0.32, this.ctx.currentTime, 0.4);
    this.musicTimer = setInterval(() => this.schedule(), 100);
    this.schedule();
  }

  stopMusic(): void {
    if (this.musicTimer) clearInterval(this.musicTimer);
    this.musicTimer = null;
    if (this.ctx && this.musicBus) this.musicBus.gain.setTargetAtTime(0.0001, this.ctx.currentTime, 0.15);
  }

  private schedule(): void {
    const ctx = this.ctx;
    if (!ctx || !this.musicBus) return;
    const spb = 60 / SoundEngine.BPM;
    while (this.nextBeat < ctx.currentTime + 0.3) {
      const t = this.nextBeat;
      const bar = Math.floor(this.beat / 4) % SoundEngine.CHORDS.length;
      const inBar = this.beat % 4;
      if (inBar === 0) {
        for (const f of SoundEngine.CHORDS[bar]) {
          this.tone(f, t, spb * 4, 'triangle', 0.07, undefined, this.musicBus);
          this.tone(f * 1.003, t, spb * 4, 'sine', 0.05, undefined, this.musicBus);
        }
        this.tone(SoundEngine.CHORDS[bar][0] / 2, t, spb * 2, 'sine', 0.18, undefined, this.musicBus);
      }
      if (inBar === 0 || inBar === 2) this.tone(110, t, 0.25, 'sine', 0.4, 45, this.musicBus);
      this.noiseBurst(t + spb / 2, 0.04, 0.06, 'highpass', 7000, this.musicBus);
      if (inBar === 1 || inBar === 3) this.noiseBurst(t, 0.12, 0.08, 'bandpass', 1800, this.musicBus);
      this.nextBeat += spb;
      this.beat++;
    }
  }
}

export const sound = new SoundEngine();
