import { useEffect, useState } from 'react'
import UploadDropzone from '../components/UploadDropzone.jsx'
import ScanningOverlay from '../components/ScanningOverlay.jsx'
import AnalysisResult from '../components/AnalysisResult.jsx'
import BatchResults from '../components/BatchResults.jsx'
import { analyzeImage, analyzeBatch, ApiError } from '../api.js'

export default function AnalyzePage() {
  const [mode, setMode] = useState('single') // 'single' | 'batch'

  // single-image state (unchanged behavior from before batch support)
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  // batch state
  const [batchFiles, setBatchFiles] = useState([])
  const [batchPreviewUrls, setBatchPreviewUrls] = useState([])
  const [batchItems, setBatchItems] = useState(null)

  useEffect(() => {
    if (!file) return
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (batchFiles.length === 0) return
    const urls = batchFiles.map((f) => URL.createObjectURL(f))
    setBatchPreviewUrls(urls)
    return () => urls.forEach((u) => URL.revokeObjectURL(u))
  }, [batchFiles])

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

  async function handleFilesSelected(selected) {
    setBatchFiles(selected)
    setBatchItems(null)
    setError(null)
    setStatus('loading')
    try {
      const data = await analyzeBatch(selected)
      // Backend processes files in submission order, so index-match them
      // to fresh local preview URLs built from that same ordered list.
      const items = data.results.map((r, idx) => ({ ...r, previewUrl: URL.createObjectURL(selected[idx]) }))
      setBatchItems(items)
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
    setBatchFiles([])
    setBatchItems(null)
    setStatus('idle')
  }

  function switchMode(next) {
    reset()
    setMode(next)
  }

  return (
    <div className="page analyze-page">
      <div className="page-header">
        <h1>Inspect an image</h1>
        <p className="page-sub">Upload a photo to check sharpness, exposure, noise, and structural defects.</p>
      </div>

      {status === 'idle' && (
        <>
          <div className="view-toggle mode-toggle">
            <button className={mode === 'single' ? 'toggle-btn active' : 'toggle-btn'} onClick={() => switchMode('single')}>
              Single image
            </button>
            <button className={mode === 'batch' ? 'toggle-btn active' : 'toggle-btn'} onClick={() => switchMode('batch')}>
              Batch
            </button>
          </div>
          {mode === 'single' ? (
            <UploadDropzone onFileSelected={handleFileSelected} />
          ) : (
            <UploadDropzone multiple onFilesSelected={handleFilesSelected} onFileSelected={(f) => handleFilesSelected([f])} />
          )}
        </>
      )}

      {status === 'loading' && mode === 'single' && (
        <div className="analyze-loading-layout">
          <ScanningOverlay imagePreviewUrl={previewUrl} />
        </div>
      )}

      {status === 'loading' && mode === 'batch' && (
        <div className="batch-loading reg-frame">
          <span className="corner-bl" /><span className="corner-br" />
          <div className="eyebrow">Analyzing {batchFiles.length} images…</div>
          <div className="batch-loading-thumbs">
            {batchPreviewUrls.map((u, idx) => (
              <img key={idx} src={u} alt="" className="batch-thumb" />
            ))}
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="error-panel reg-frame">
          <span className="corner-bl" /><span className="corner-br" />
          <div className="eyebrow">Analysis failed</div>
          <p>{error}</p>
          <button className="btn-primary" onClick={reset}>Try again</button>
        </div>
      )}

      {status === 'success' && mode === 'single' && result && (
        <div className="fade-in">
          <AnalysisResult result={result} imagePreviewUrl={previewUrl} />
          <div className="analyze-actions">
            <button className="btn-secondary" onClick={reset}>Inspect another image</button>
          </div>
        </div>
      )}

      {status === 'success' && mode === 'batch' && batchItems && (
        <div className="fade-in">
          <BatchResults items={batchItems} />
          <div className="analyze-actions">
            <button className="btn-secondary" onClick={reset}>Inspect more images</button>
          </div>
        </div>
      )}
    </div>
  )
}
