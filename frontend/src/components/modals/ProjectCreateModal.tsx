import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { X, Upload, Plus, AlertCircle, Loader2 } from "lucide-react";
import { useUIStore } from "../../store/uiStore";
import { api } from "../../services/api";

export function ProjectCreateModal({ onCreated }: { onCreated?: () => void }) {
  const { isCreateProjectOpen, setIsCreateProjectOpen } = useUIStore();
  const navigate = useNavigate();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [targetMetric, setTargetMetric] = useState("");
  const [startRunImmediately, setStartRunImmediately] = useState(true);

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isCreateProjectOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError("Please provide a project name.");
      return;
    }
    if (!file) {
      setError("Please select a tabular CSV dataset to upload.");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only tabular .csv files are supported.");
      return;
    }
    if (startRunImmediately && !targetColumn.trim()) {
      setError(
        "Target column is required to start a run immediately. Please specify a target column or uncheck 'Start initial run immediately'."
      );
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      // 1. Create project
      const proj = await api.createProject({
        name: name.trim(),
        description: description.trim() || undefined,
      });

      // 2. Upload dataset
      const ds = await api.uploadDataset(proj.id, file);

      // 3. Optionally start initial run
      if (startRunImmediately && targetColumn.trim()) {
        await api.triggerRun(proj.id, {
          target_column: targetColumn.trim(),
          target_metric: targetMetric.trim() || undefined,
          dataset_version: ds.version,
        });
      }

      setIsCreateProjectOpen(false);
      if (onCreated) onCreated();
      navigate(`/projects/${proj.id}`);
    } catch (err: any) {
      setError(err.message || "Failed to create project");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-[#0f172a] border border-slate-700 rounded-xl w-full max-w-xl flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        {/* Header */}
        <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-900/60">
          <div className="flex items-center gap-2">
            <Plus className="w-5 h-5 text-sky-400" />
            <h3 className="font-semibold text-slate-100 text-sm">Create New Project</h3>
          </div>
          <button
            onClick={() => setIsCreateProjectOpen(false)}
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

          {/* Project Name */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">Project Name *</label>
            <input
              type="text"
              required
              placeholder="e.g. Titanic Survival Predictor"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-sans"
            />
          </div>

          {/* Description */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">Description (Optional)</label>
            <textarea
              rows={2}
              placeholder="Brief context or goals for this tabular ML project"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-sans resize-none"
            />
          </div>

          {/* Dataset Upload */}
          <div>
            <label className="block text-slate-300 font-medium mb-1">Tabular Dataset (.CSV) *</label>
            <div className="border-2 border-dashed border-slate-700 rounded-lg p-4 text-center hover:border-slate-600 transition bg-slate-900/40">
              <input
                type="file"
                accept=".csv"
                id="dataset-upload"
                className="hidden"
                onChange={(e) => {
                  if (e.target.files && e.target.files[0]) {
                    setFile(e.target.files[0]);
                  }
                }}
              />
              <label htmlFor="dataset-upload" className="cursor-pointer flex flex-col items-center gap-1.5">
                <Upload className="w-6 h-6 text-sky-400" />
                <span className="text-slate-300 font-medium">
                  {file ? file.name : "Click to select CSV file"}
                </span>
                <span className="text-[11px] text-slate-500 font-mono">
                  {file ? `${(file.size / 1024).toFixed(1)} KB` : "Supports standard comma-separated tabular files"}
                </span>
              </label>
            </div>
          </div>

          {/* Target & Metric */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="block text-slate-300 font-medium mb-1">
                Target Column <span className="text-slate-500 font-normal">(can set later)</span>
              </label>
              <input
                type="text"
                placeholder="e.g. survived, price, label"
                value={targetColumn}
                onChange={(e) => setTargetColumn(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
              />
            </div>
            <div>
              <label className="block text-slate-300 font-medium mb-1">
                Target Metric <span className="text-slate-500 font-normal">(optional auto-select)</span>
              </label>
              <input
                type="text"
                placeholder="e.g. f1, accuracy, rmse, r2"
                value={targetMetric}
                onChange={(e) => setTargetMetric(e.target.value)}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 placeholder-slate-500 focus:outline-none focus:border-sky-500 font-mono"
              />
            </div>
          </div>

          {/* Options */}
          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="start-run-checkbox"
                checked={startRunImmediately}
                onChange={(e) => setStartRunImmediately(e.target.checked)}
                className="rounded border-slate-700 text-sky-500 focus:ring-0 bg-slate-900"
              />
              <label htmlFor="start-run-checkbox" className="text-slate-300 select-none">
                Start initial run immediately if target provided
              </label>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="pt-4 border-t border-slate-800 flex items-center justify-end gap-2.5">
            <button
              type="button"
              onClick={() => setIsCreateProjectOpen(false)}
              className="px-4 py-2 rounded-lg border border-slate-700 text-slate-300 hover:bg-slate-800 transition text-xs font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold transition text-xs disabled:opacity-50"
            >
              {isLoading && <Loader2 className="w-3.5 h-3.5 animate-spin" />}
              <span>{isLoading ? "Creating & Uploading..." : "Create Project"}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
