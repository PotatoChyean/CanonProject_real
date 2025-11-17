"use client"

import { useState } from "react"
import { Play, Square } from "lucide-react"

export function LiveCamera({ setIsProcessing }: any) {
  const [isRunning, setIsRunning] = useState(false)
  const [frameCount, setFrameCount] = useState(0)

  const handleStartDetection = () => {
    setIsProcessing(true)
    setIsRunning(true)
    // Simulate frame updates
    const interval = setInterval(() => {
      setFrameCount((prev) => (prev < 150 ? prev + 1 : 0))
    }, 67) // ~15 FPS

    setTimeout(() => {
      setIsRunning(false)
      setIsProcessing(false)
      clearInterval(interval)
      setFrameCount(0)
    }, 10000)
  }

  const handleStop = () => {
    setIsRunning(false)
    setIsProcessing(false)
    setFrameCount(0)
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Camera Feed Display */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden shadow-xl">
        <div className="aspect-video bg-gradient-to-br from-slate-900 to-slate-950 flex items-center justify-center relative">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-full h-full bg-[linear-gradient(45deg,transparent_30%,rgba(255,255,255,.1)_50%,transparent_70%)] opacity-0 animate-pulse"></div>
          </div>
          <div className="text-center z-10">
            <div className="w-24 h-24 rounded-full border-4 border-blue-500/30 mx-auto mb-4 flex items-center justify-center">
              <div className="w-20 h-20 rounded-full border-4 border-blue-500/50"></div>
            </div>
            <p className="text-slate-300 font-medium">
              {isRunning ? `Detecting... (Frame ${frameCount}/150)` : "Camera Ready - 1280×800"}
            </p>
          </div>
        </div>

        {/* Camera Info Bar */}
        <div className="bg-slate-900 border-t border-slate-700 px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-slate-400">
            <span>Resolution: 1280×800</span>
            <span>•</span>
            <span>FPS: 15</span>
            <span>•</span>
            <span>Status: {isRunning ? "Recording" : "Idle"}</span>
          </div>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="flex gap-4">
        {!isRunning ? (
          <button
            onClick={handleStartDetection}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-500 hover:from-green-700 hover:to-emerald-600 text-white font-semibold rounded-lg transition-all shadow-lg shadow-green-500/30"
          >
            <Play className="w-5 h-5" />
            Start Detection
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-red-600 to-pink-500 hover:from-red-700 hover:to-pink-600 text-white font-semibold rounded-lg transition-all shadow-lg shadow-red-500/30"
          >
            <Square className="w-5 h-5" />
            Stop Detection
          </button>
        )}
      </div>

      {/* Analysis Indicator */}
      {isRunning && (
        <div className="bg-blue-900/30 border border-blue-700/50 rounded-lg p-4 text-center">
          <p className="text-blue-200 font-medium animate-pulse">Analyzing frames, please wait...</p>
        </div>
      )}
    </div>
  )
}
