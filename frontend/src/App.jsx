import { Routes, Route } from 'react-router-dom'
import NavBar from './components/NavBar.jsx'
import AnalyzePage from './pages/AnalyzePage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import AnalysisDetailPage from './pages/AnalysisDetailPage.jsx'

export default function App() {
  return (
    <div className="app-shell">
      <NavBar />
      <main className="app-main">
        <Routes>
          <Route path="/" element={<AnalyzePage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/history/:id" element={<AnalysisDetailPage />} />
        </Routes>
      </main>
    </div>
  )
}
