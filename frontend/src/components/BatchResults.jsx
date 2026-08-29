import AnalysisResult from './AnalysisResult.jsx'

export default function BatchResults({ items }) {
  const succeeded = items.filter((i) => i.status === 'ok').length
  const failed = items.length - succeeded

  return (
    <div className="batch-results">
      <div className="batch-summary mono">
        <span className="batch-summary-total">{items.length} images analyzed</span>
        <span className="batch-summary-ok">{succeeded} succeeded</span>
        {failed > 0 && <span className="batch-summary-failed">{failed} failed</span>}
      </div>

      {items.map((item, idx) => (
        <div className="batch-item" key={idx}>
          <div className="batch-item-head">
            <span className="eyebrow">{item.filename}</span>
            {item.status === 'error' && <span className="status-chip status-DEFECTIVE">FAILED</span>}
          </div>

          {item.status === 'ok' ? (
            <AnalysisResult result={item.analysis} imagePreviewUrl={item.previewUrl} />
          ) : (
            <div className="error-panel reg-frame batch-item-error">
              <span className="corner-bl" /><span className="corner-br" />
              <p>{item.error}</p>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
