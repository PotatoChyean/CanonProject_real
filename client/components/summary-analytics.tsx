"use client"

import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { Download } from "lucide-react"

export function SummaryAnalytics({ results }: any) {
  const passCount = results.filter((r: any) => r.status === "PASS").length
  const failCount = results.filter((r: any) => r.status === "FAIL").length
  const totalCount = results.length
  const passRate = totalCount > 0 ? Math.round((passCount / totalCount) * 100) : 0

  const pieData = [
    { name: "Pass", value: passCount, fill: "#10b981" },
    { name: "Fail", value: failCount, fill: "#ef4444" },
  ]

  const barData = [
    { name: "Jan", pass: 45, fail: 12 },
    { name: "Feb", pass: 52, fail: 18 },
    { name: "Mar", pass: 48, fail: 14 },
    { name: "Apr", pass: 61, fail: 9 },
    { name: "May", pass: 55, fail: 16 },
  ]

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Summary & Analytics</h2>
        <button className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-colors">
          <Download className="w-4 h-4" />
          Download Report
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard label="Total Analyzed" value={totalCount} color="bg-blue-600" />
        <StatCard label="Pass" value={passCount} color="bg-emerald-600" />
        <StatCard label="Fail" value={failCount} color="bg-red-600" />
        <StatCard label="Pass Rate" value={`${passRate}%`} color="bg-cyan-600" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Pie Chart */}
        {totalCount > 0 && (
          <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Results Distribution</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Bar Chart */}
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Monthly Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={barData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #475569" }} />
              <Legend />
              <Bar dataKey="pass" fill="#10b981" />
              <Bar dataKey="fail" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

function StatCard({ label, value, color }: { label: string; value: string | number; color: string }) {
  return (
    <div className={`${color} rounded-lg p-6 text-white`}>
      <p className="text-sm font-medium opacity-90">{label}</p>
      <p className="text-3xl font-bold mt-2">{value}</p>
    </div>
  )
}
