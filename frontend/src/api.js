const BASE = import.meta.env.VITE_API_BASE || ''

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function handleResponse(res) {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || detail
    } catch {
      // response wasn't JSON — fall back to statusText
    }
    throw new ApiError(detail, res.status, detail)
  }
  return res.json()
}

export async function analyzeImage(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/api/v1/analyze`, { method: 'POST', body: form })
  return handleResponse(res)
}

export async function listAnalyses({ limit = 20, offset = 0, qualityLabel = null } = {}) {
  const params = new URLSearchParams({ limit, offset })
  if (qualityLabel) params.set('quality_label', qualityLabel)
  const res = await fetch(`${BASE}/api/v1/analyses?${params}`)
  return handleResponse(res)
}

export async function getAnalysis(id) {
  const res = await fetch(`${BASE}/api/v1/analyses/${id}`)
  return handleResponse(res)
}

export function getAnalysisImageUrl(id) {
  return `${BASE}/api/v1/analyses/${id}/image`
}

export async function getHealth() {
  const res = await fetch(`${BASE}/health`)
  return handleResponse(res)
}

export { ApiError }
