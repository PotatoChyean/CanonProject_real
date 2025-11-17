"use client"

import { Clock } from "lucide-react"

export function TopBar({ isProcessing }: { isProcessing: boolean }) {
  return (
    <div className="bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <div
            className={`w-3 h-3 rounded-full transition-colors ${isProcessing ? "bg-amber-500 animate-pulse" : "bg-emerald-500"}`}
          ></div>
          <span className="text-sm font-medium text-slate-300">
            Status: <span className="text-white">{isProcessing ? "Processing..." : "Idle"}</span>
          </span>
        </div>
      </div>
      <div className="flex items-center gap-4 text-sm text-slate-400">
        <div className="flex items-center gap-2">
          <Clock className="w-4 h-4" />
          <span>Last run: 2 hours ago</span>
        </div>
      </div>
    </div>
  )
}
