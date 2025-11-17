"use client"

import type React from "react"

import { useState } from "react"
import { Upload, File } from "lucide-react"

export function ImageUpload({ setIsProcessing, setResults }: any) {
  const [files, setFiles] = useState<File[]>([])
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = Array.from(e.dataTransfer.files)
    setFiles((prev) => [...prev, ...droppedFiles])
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      setFiles((prev) => [...prev, ...selectedFiles])
    }
  }

  const handleStartAnalysis = async () => {
    if (files.length === 0) return

    setIsProcessing(true)
    // Simulate API call
    await new Promise((resolve) => setTimeout(resolve, 3000))

    // Generate mock results
    const mockResults = files.map((file, index) => ({
      id: index,
      name: file.name,
      status: Math.random() > 0.3 ? "PASS" : "FAIL",
      reason:
        Math.random() > 0.3
          ? null
          : ["Blurred region", "Misalignment", "Defect detected"][Math.floor(Math.random() * 3)],
      confidence: Math.floor(Math.random() * 40 + 60),
    }))

    setResults(mockResults)
    setIsProcessing(false)
  }

  const handleRemoveFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Drag and Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-12 text-center transition-all ${
          isDragging ? "border-blue-500 bg-blue-500/10" : "border-slate-600 bg-slate-800/30 hover:border-slate-500"
        }`}
      >
        <Upload className="w-12 h-12 text-slate-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold text-white mb-2">Upload Image Folder</h3>
        <p className="text-slate-400 mb-6">Drag and drop your images here or click below to select files</p>
        <label className="inline-block">
          <input type="file" multiple accept="image/*" onChange={handleFileSelect} className="hidden" />
          <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors">
            Select Folder
          </button>
        </label>
      </div>

      {/* File List */}
      {files.length > 0 && (
        <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6">
          <h4 className="text-sm font-semibold text-white mb-4">Selected Files ({files.length})</h4>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {files.map((file, index) => (
              <div
                key={index}
                className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-700"
              >
                <div className="flex items-center gap-3">
                  <File className="w-4 h-4 text-blue-400" />
                  <span className="text-sm text-slate-300">{file.name}</span>
                </div>
                <button
                  onClick={() => handleRemoveFile(index)}
                  className="text-slate-400 hover:text-red-400 transition-colors"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Start Analysis Button */}
      {files.length > 0 && (
        <button
          onClick={handleStartAnalysis}
          className="w-full py-3 bg-gradient-to-r from-blue-600 to-cyan-500 hover:from-blue-700 hover:to-cyan-600 text-white font-semibold rounded-lg transition-all shadow-lg shadow-blue-500/30"
        >
          Start Analysis
        </button>
      )}
    </div>
  )
}
