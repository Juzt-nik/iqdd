import { useState } from 'react'

const FEATURE_LABELS = {
  laplacian_variance: 'Sharpness (Laplacian var.)',
  tenengrad: 'Sharpness (gradient energy)',
  mean_luminance: 'Mean luminance',
  underexposed_frac: 'Underexposed pixels',
  overexposed_frac: 'Overexposed / clipped pixels',
  noise_sigma: 'Noise σ estimate',
  luminance_std: 'Contrast (luminance σ)',
  dynamic_range: 'Dynamic range (p99–p1)',
  colorfulness: 'Colorfulness',
  saturation_mean: 'Mean saturation',
  blockiness: 'Compression blockiness',
  entropy: 'Entropy (bits)',
}

const FRACTION_KEYS = new Set(['underexposed_frac', 'overexposed_frac'])

function formatFeatureValue(key, value) {
  if (FRACTION_KEYS.has(key)) return `${(value * 100).toFixed(1)}%`
  return Number.isInteger(value) ? String(value) : value.toFixed(2)
}

export default function AnalysisResult({ result, imagePreviewUrl }) {
  const [showHeatmap, setShowHeatmap] = useState(false)
  const { quality_score, quality_label, issues, features, heatmap_png_base64, model_versions } = result

  return (
    <div className="result-grid">
      <div className="result-image-col">
        <div className="reg-frame image-frame active">
          <span className="corner-bl" />
          <span className="corner-br" />
          <img
            src={showHeatmap && heatmap_png_base64 ? `data:image/png;base64,${heatmap_png_base64}` : imagePreviewUrl}
            alt={showHeatmap ? 'Defect heatmap overlay' : 'Uploaded image'}
            className="result-image"
          />
        </div>
        {heatmap_png_base64 && (
          <div className="view-toggle">
            <button
              className={!showHeatmap ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setShowHeatmap(false)}
            >
              Original
            </button>
            <button
              className={showHeatmap ? 'toggle-btn active' : 'toggle-btn'}
              onClick={() => setShowHeatmap(true)}
            >
              Defect view
            </button>
          </div>
        )}
      </div>

      <div className="result-data-col">
        <section className="score-panel">
          <div className="eyebrow">Quality score</div>
          <div className="score-value mono">{quality_score.toFixed(1)}</div>
          <span className={`status-chip status-${quality_label}`}>{quality_label}</span>
        </section>

        <section>
          <div className="eyebrow">Detected issues</div>
          {issues.length === 0 ? (
            <div className="no-issues">No issues detected above the confidence threshold.</div>
          ) : (
            <ul className="issue-list">
              {issues.map((issue) => (
                <li key={issue.type} className={`issue-item issue-border-${issue.severity}`}>
                  <div className="issue-item-head">
                    <span className={`issue-type severity-${issue.severity}`}>{issue.type}</span>
                    <span className="issue-meta mono">
                      {issue.severity} · {(issue.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="issue-explanation">{issue.explanation}</div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section>
          <div className="eyebrow">Image statistics</div>
          <dl className="stats-grid mono">
            {Object.entries(FEATURE_LABELS).map(([key, label]) => (
              features[key] !== undefined && (
                <div className="stat-row" key={key}>
                  <dt>{label}</dt>
                  <dd>{formatFeatureValue(key, features[key])}</dd>
                </div>
              )
            ))}
          </dl>
        </section>

        <section className="model-versions">
          <div className="eyebrow">Models used</div>
          <div className="model-badges mono">
            {Object.entries(model_versions).map(([name, active]) => (
              <span key={name} className={active ? 'model-badge active' : 'model-badge'}>
                {name}
              </span>
            ))}
          </div>
        </section>
      </div>
    </div>
  )
}
