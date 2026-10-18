import { memo } from 'react';
import type { GlyphHand } from '../config/poses';
import { glyphGeometry } from '../render/glyph';

interface Props {
  hands: GlyphHand[];
  className?: string;
  /** Silhouette colour (defaults to currentColor). */
  fill?: string;
  /** Colour used to separate overlapping fingers — match the background. */
  outline?: string;
  title?: string;
}

/** Hand-pose illustration generated from the same model the classifier is tested on. */
export const HandGlyph = memo(function HandGlyph({ hands, className, fill = 'currentColor', outline = '#141412', title }: Props) {
  const g = glyphGeometry(hands);
  const S = 100;
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className={className} role={title ? 'img' : undefined} aria-hidden={title ? undefined : true}>
      {title ? <title>{title}</title> : null}
      <g strokeLinecap="round" strokeLinejoin="round">
        {g.palms.map((palm, i) => (
          <polygon key={`p${i}`} points={palm.map(([x, y]) => `${x * S},${y * S}`).join(' ')} fill={fill} stroke={fill} strokeWidth={5} />
        ))}
        {g.segments.map((s, i) => (
          <g key={i}>
            <line x1={s.x1 * S} y1={s.y1 * S} x2={s.x2 * S} y2={s.y2 * S} stroke={outline} strokeWidth={(s.w + 0.016) * S} />
            <line x1={s.x1 * S} y1={s.y1 * S} x2={s.x2 * S} y2={s.y2 * S} stroke={fill} strokeWidth={s.w * S} />
          </g>
        ))}
      </g>
    </svg>
  );
});
