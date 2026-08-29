export default function ScanningOverlay({ imagePreviewUrl }) {
  return (
    <div className="reg-frame image-frame active scanning-frame">
      <span className="corner-bl" />
      <span className="corner-br" />
      <img src={imagePreviewUrl} alt="Analyzing" className="result-image scanning-image" />
      <div className="scan-line" />
      <div className="scanning-label mono">ANALYZING…</div>
    </div>
  )
}
