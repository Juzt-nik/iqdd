import { useCallback, useRef, useState } from 'react'

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/webp']

export default function UploadDropzone({ onFileSelected, onFilesSelected, disabled, multiple = false }) {
  const [isDragging, setIsDragging] = useState(false)
  const [rejectReason, setRejectReason] = useState(null)
  const inputRef = useRef(null)

  const validateAndEmit = useCallback((fileList) => {
    const files = Array.from(fileList || [])
    if (files.length === 0) return
    const invalid = files.find((f) => !ACCEPTED_TYPES.includes(f.type))
    if (invalid) {
      setRejectReason(`"${invalid.type || 'unknown type'}" isn't a supported image format. Use JPEG, PNG, BMP, or WEBP.`)
      return
    }
    setRejectReason(null)
    if (files.length > 1 && onFilesSelected) {
      onFilesSelected(files)
    } else {
      onFileSelected(files[0])
    }
  }, [onFileSelected, onFilesSelected])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    if (disabled) return
    validateAndEmit(e.dataTransfer.files)
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
        aria-label={multiple ? 'Upload one or more images to inspect' : 'Upload an image to inspect'}
      >
        <span className="corner-bl" />
        <span className="corner-br" />
        <div className="dropzone-content">
          <div className="dropzone-icon">⌖</div>
          <div className="dropzone-title">
            {multiple ? 'Drop images to inspect' : 'Drop an image to inspect'}
          </div>
          <div className="dropzone-sub">
            {multiple
              ? `or click to browse — select multiple · up to ${10} files · JPEG, PNG, BMP, WEBP · 15MB each`
              : 'or click to browse — JPEG, PNG, BMP, WEBP · up to 15MB'}
          </div>
        </div>
        <input
          ref={inputRef}
          type="file"
          multiple={multiple}
          accept={ACCEPTED_TYPES.join(',')}
          style={{ display: 'none' }}
          onChange={(e) => validateAndEmit(e.target.files)}
        />
      </div>
      {rejectReason && <div className="dropzone-error">{rejectReason}</div>}
    </div>
  )
}
