import { useState, useEffect } from "react";
import { X, Play, AlertCircle, Loader2 } from "lucide-react";
import { useUIStore } from "../../store/uiStore";
import { api } from "../../services/api";
import { Dataset } from "../../types";

interface TriggerRunModalProps {
  projectId: string;
  datasets: Dataset[];
  onTriggered?: (runId: string) => void;
}

export function TriggerRunModal({ projectId, datasets, onTriggered }: TriggerRunModalProps) {
  const { isTriggerRunOpen, setIsTriggerRunOpen } = useUIStore();

  const [datasetVersion, setDatasetVersion] = useState<string>(
    datasets.length > 0 ? datasets[0].version : "dataset_v1"
  );
  const [targetColumn, setTargetColumn] = useState("");
  const [targetMetric, setTargetMetric] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (datasets.length > 0 && !datasets.some((d) => d.version === datasetVersion)) {
      setDatasetVersion(datasets[0].version);
    }
  }, [datasets, datasetVersion]);

  if (!isTriggerRunOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (datasets.length === 0) {
      setError("No datasets available. Please upload a dataset in the Dataset tab first.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const run = await api.triggerRun(projectId, {
        target_column: targetColumn.trim() || undefined,
        target_metric: targetMetric.trim() || undefined,
        dataset_version: datasetVersion,
      });

      setIsTriggerRunOpen(false);
      if (onTriggered) onTriggered(run.id);
    } catch (err: any) {
      setError(err.message || "Failed to trigger workflow run");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0f172a] border border-slate-700 rounded-xl w-full max-w-lg flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-2">
            <Play className="w-4 h-4 text-sky-400 fill-current" />
            <h3 className="font-semibold text-slate-100 text-sm">Start Workflow Run</h3>
          </div>
          <button
            onClick={() => setIsTriggerRunOpen(false)}
            className="p-1 rounded-md text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="p-6 space-y-4 text-xs">
          {error && (
            <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg flex items-center gap-2 text-rose-400">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Dataset Version */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">Dataset Version</label>
            <select
              value={datasetVersion}
              onChange={(e) => setDatasetVersion(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 font-mono focus:outline-none focus:border-sky-500"
            >
              {datasets.map((ds) => (
                <option key={ds.id} value={ds.version}>
                  {ds.version} — {ds.filename} ({ds.row_count} rows, {ds.col_count} cols)
                </option>
              ))}
            </select>
          </div>

          {/* Target Column */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">
              Target Column Name <span className="text-slate-500 font-normal">(Leave empty to trigger NEEDS_INPUT)</span>
            </label>
            <input
              type="text"
              placeholder="e.g. survived, price, label"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
            />
          </div>

          {/* Target Metric */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">
              Target Metric <span className="text-slate-500 font-normal">(Optional: f1, accuracy, rmse, mae)</span>
            </label>
            <input
              type="text"
              placeholder="Auto-detect based on task type"
              value={targetMetric}
              onChange={(e) => setTargetMetric(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
            />
          </div>

          {/* Footer Actions */}
          <div className="pt-4 border-t border-slate-800 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setIsTriggerRunOpen(false)}
              className="px-4 py-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 transition font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading || datasets.length === 0}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold transition disabled:opacity-50"
            >
              {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>{isLoading ? "Starting Pipeline..." : "Execute Run"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
