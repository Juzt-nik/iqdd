import { useEffect, useState } from 'react'
import UploadDropzone from '../components/UploadDropzone.jsx'
import ScanningOverlay from '../components/ScanningOverlay.jsx'
import AnalysisResult from '../components/AnalysisResult.jsx'
import { analyzeImage, ApiError } from '../api.js'

export default function AnalyzePage() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function handleFileSelected(selected) {
    setFile(selected)
    setResult(null)
    setError(null)
    setStatus('loading')
    try {
      const data = await analyzeImage(selected)
      setResult(data)
      setStatus('success')
    } catch (err) {
      const message = err instanceof ApiError ? err.detail : 'Could not reach the analysis service. Check the backend is running.'
      setError(message)
      setStatus('error')
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    setStatus('idle')
  }

  return (
    <div className="page analyze-page">
      <div className="page-header">
        <h1>Inspect an image</h1>
        <p className="page-sub">Upload a photo to check sharpness, exposure, noise, and structural defects.</p>
      </div>

      {status === 'idle' && (
        <UploadDropzone onFileSelected={handleFileSelected} />
      )}

      {status === 'loading' && (
        <div className="analyze-loading-layout">
          <ScanningOverlay imagePreviewUrl={previewUrl} />
        </div>
      )}

      {status === 'error' && (
        <div className="error-panel reg-frame">
          <span className="corner-bl" /><span className="corner-br" />
          <div className="eyebrow">Analysis failed</div>
          <p>{error}</p>
          <button className="btn-primary" onClick={reset}>Try another image</button>
        </div>
      )}

      {status === 'success' && result && (
        <div className="fade-in">
          <AnalysisResult result={result} imagePreviewUrl={previewUrl} />
          <div className="analyze-actions">
            <button className="btn-secondary" onClick={reset}>Inspect another image</button>
          </div>
        </div>
      )}
    </div>
  )
}
