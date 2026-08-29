import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { listAnalyses, ApiError } from '../api.js'

const FILTERS = ['ALL', 'ACCEPTABLE', 'DEGRADED', 'DEFECTIVE']

export default function HistoryPage() {
  const [data, setData] = useState(null)
  const [filter, setFilter] = useState('ALL')
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    listAnalyses({ limit: 50, qualityLabel: filter === 'ALL' ? null : filter })
      .then((d) => { if (!cancelled) { setData(d); setStatus('success') } })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof ApiError ? err.detail : 'Could not load history.')
        setStatus('error')
      })
    return () => { cancelled = true }
  }, [filter])

  return (
    <div className="page history-page">
      <div className="page-header">
        <h1>History</h1>
        <p className="page-sub">Past inspections, most recent first.</p>
      </div>

      <div className="filter-row mono">
        {FILTERS.map((f) => (
          <button
            key={f}
            className={filter === f ? 'filter-chip active' : 'filter-chip'}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      {status === 'loading' && <div className="loading-text mono">Loading…</div>}
      {status === 'error' && <div className="error-text">{error}</div>}

      {status === 'success' && data.results.length === 0 && (
        <div className="empty-state">
          <div className="eyebrow">Nothing here yet</div>
          <p>Run an inspection to see it appear in this history.</p>
          <Link to="/" className="btn-primary">Inspect an image</Link>
        </div>
      )}

      {status === 'success' && data.results.length > 0 && (
        <div className="contact-sheet">
          {data.results.map((item) => (
            <Link to={`/history/${item.id}`} key={item.id} className="sheet-item">
              <div className="sheet-item-status">
                <span className={`status-chip status-${item.quality_label}`}>{item.quality_label}</span>
              </div>
              <div className="sheet-item-name mono">{item.original_filename}</div>
              <div className="sheet-item-score mono">{item.quality_score.toFixed(1)}</div>
              {item.issue_types.length > 0 && (
                <div className="sheet-item-issues mono">{item.issue_types.join(', ')}</div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}
