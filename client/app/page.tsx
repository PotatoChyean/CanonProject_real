"use client"

import { useState } from "react"
import { Sidebar } from "@/components/sidebar"
import { TopBar } from "@/components/top-bar"
import { Navigation } from "@/components/navigation"
import { ImageUpload } from "@/components/image-upload"
import { LiveCamera } from "@/components/live-camera"
import { ResultsGrid } from "@/components/results-grid"
import { SummaryAnalytics } from "@/components/summary-analytics"

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<"upload" | "live" | "results" | "summary">("upload")
  const [isProcessing, setIsProcessing] = useState(false)
  const [results, setResults] = useState<any[]>([])
  const [isCollapsed, setIsCollapsed] = useState(false)
  const paddingClass = isCollapsed ? 'pl-16' : 'pl-64';

  return (
    <div className="flex h-screen bg-slate-950 relative">
      <Sidebar isCollapsed={isCollapsed} setIsCollapsed={setIsCollapsed} />
      <div className={`flex-1 flex flex-col transition-all duration-300 ${paddingClass}`}>
        <TopBar isProcessing={isProcessing} />
        <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="flex-1 overflow-auto bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900 p-6">
          {activeTab === "upload" && <ImageUpload setIsProcessing={setIsProcessing} setResults={setResults} />}
          {activeTab === "live" && <LiveCamera setIsProcessing={setIsProcessing} />}
          {activeTab === "results" && <ResultsGrid results={results} />}
          {activeTab === "summary" && <SummaryAnalytics results={results} />}
        </main>
      </div>
    </div>
  )
}
