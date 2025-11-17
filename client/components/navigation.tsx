"use client"
import type React from "react"
import { Upload, Camera, BarChart3, TrendingUp } from "lucide-react"

export function Navigation({ activeTab, setActiveTab }: any) {
  const tabs = [
    { 
        id: "upload", 
        label: "업로드", 
        icon: <Upload className="w-5 h-5" /> 
    },
    { 
        id: "live", 
        label: "카메라", 
        icon: <Camera className="w-5 h-5" /> // BarChart3 대신 Camera로 변경했습니다.
    },
    { 
        id: "results", 
        label: "결과", 
        icon: <BarChart3 className="w-5 h-5" /> // Lucide Icon으로 통일
    },
    { 
        id: "analytics", 
        label: "분석", 
        icon: <TrendingUp className="w-5 h-5" /> 
    },
  ]

  return (
    <div className="bg-slate-900 border-b border-slate-700 px-6 py-4 flex gap-6">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => setActiveTab(tab.id)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
            activeTab === tab.id
              ? "bg-blue-600 text-white shadow-lg shadow-blue-500/30"
              : "text-slate-300 hover:text-white hover:bg-slate-800"
          }`}
        >
          <span className="mr-2">{tab.icon}</span>
          {tab.label}
        </button>
      ))}
    </div>
  )
}
