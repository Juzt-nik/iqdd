import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import AnalysisResult from '../components/AnalysisResult.jsx'
import { getAnalysis, getAnalysisImageUrl, ApiError } from '../api.js'

export default function AnalysisDetailPage() {
  const { id } = useParams()
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState('loading')
  const [error, setError] = useState(null)

  useEffect(() => {
    let cancelled = false
    setStatus('loading')
    getAnalysis(id)
      .then((d) => { if (!cancelled) { setResult(d); setStatus('success') } })
      .catch((err) => {
        if (cancelled) return
        setError(err instanceof ApiError ? err.detail : 'Could not load this analysis.')
        setStatus('error')
      })
    return () => { cancelled = true }
  }, [id])

  return (
    <div className="page">
      <div className="page-header">
        <Link to="/history" className="back-link mono">&larr; History</Link>
        <h1>{status === 'success' ? result.original_filename : 'Analysis'}</h1>
        {status === 'success' && (
          <p className="page-sub mono">{new Date(result.created_at).toLocaleString()}</p>
        )}
      </div>

      {status === 'loading' && <div className="loading-text mono">Loading…</div>}
      {status === 'error' && <div className="error-text">{error}</div>}
      {status === 'success' && (
        <AnalysisResult result={result} imagePreviewUrl={getAnalysisImageUrl(id)} />
      )}
    </div>
  )
}
