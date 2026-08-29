import { useCallback, useRef, useState } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']

export default function UploadDropzone({ onFileSelected, disabled }) {
  const [isDragging, setIsDragging] = useState(false)
  const [rejectReason, setRejectReason] = useState(null)
  const inputRef = useRef(null)

  const validateAndEmit = useCallback((file) => {
    if (!file) return
    if (!ACCEPTED_TYPES.includes(file.type)) {
      setRejectReason(`"${file.type || 'unknown type'}" isn't a supported image format. Use JPEG, PNG, BMP, or WEBP.`)
      return
    }
    setRejectReason(null)
    onFileSelected(file)
  }, [onFileSelected])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    validateAndEmit(file)
  }, [disabled, validateAndEmit])

  return (
    <div className="dropzone-wrap">
      <div
        className={`dropzone reg-frame ${isDragging ? 'active dragging' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={(e) => { e.preventDefault(); if (!disabled) setIsDragging(true) }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if ((e.key === 'Enter' || e.key === ' ') && !disabled) inputRef.current?.click() }}
        aria-label="Upload an image to inspect"
      >
        <span className="corner-bl" />
        <span className="corner-br" />
        <div className="dropzone-content">
          <div className="dropzone-icon">⌖</div>
          <div className="dropzone-title">Drop an image to inspect</div>
          <div className="dropzone-sub">or click to browse — JPEG, PNG, BMP, WEBP · up to 15MB</div>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          style={{ display: 'none' }}
          onChange={(e) => validateAndEmit(e.target.files?.[0])}
        />
      </div>
      {rejectReason && <div className="dropzone-error">{rejectReason}</div>}
    </div>
  )
}
