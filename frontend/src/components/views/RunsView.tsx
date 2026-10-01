import { History, Play, ArrowRight } from "lucide-react";
import { WorkflowRun } from "../../types";
import { StatusBadge } from "../common/Badge";
import { useUIStore } from "../../store/uiStore";

interface RunsViewProps {
  projectId: string;
  runs: WorkflowRun[];
  onTriggerNewRun: () => void;
}

export function RunsView({ projectId: _projectId, runs, onTriggerNewRun }: RunsViewProps) {
  const { setActiveRunId, setActiveTab } = useUIStore();

  const handleSelectRun = (runId: string) => {
    setActiveRunId(runId);
    setActiveTab("overview");
  };

  return (
    <div className="p-6 space-y-6 h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <History className="w-5 h-5 text-sky-400" />
            <h2 className="text-base font-bold text-slate-100">Workflow Runs History</h2>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Complete execution timeline of autonomous pipeline runs for this project.
          </p>
        </div>

        <button
          onClick={onTriggerNewRun}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold text-xs transition"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>New Run</span>
        </button>
      </div>

      {runs.length === 0 ? (
        <div className="p-12 text-center border border-slate-800 rounded-xl bg-slate-900/30 space-y-2">
          <History className="w-8 h-8 text-slate-600 mx-auto" />
          <div className="text-xs text-slate-400 font-medium">No runs recorded yet</div>
          <div className="text-[11px] text-slate-500 font-mono">
            Trigger a run to start autonomous modeling.
          </div>
        </div>
      ) : (
        <div className="border border-slate-800 rounded-xl overflow-hidden bg-slate-900/40">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-800/60 font-mono text-slate-400 border-b border-slate-800 select-none">
              <tr>
                <th className="p-3">Run ID</th>
                <th className="p-3">Status</th>
                <th className="p-3">Target Column</th>
                <th className="p-3">Target Metric</th>
                <th className="p-3">Dataset Version</th>
                <th className="p-3">Best Score</th>
                <th className="p-3">Created</th>
                <th className="p-3 w-10"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80 font-mono">
              {runs.map((r) => (
                <tr
                  key={r.id}
                  onClick={() => handleSelectRun(r.id)}
                  className="hover:bg-slate-800/40 cursor-pointer transition group"
                >
                  <td className="p-3 font-semibold text-sky-400 group-hover:underline">
                    {r.id}
                  </td>
                  <td className="p-3">
                    <StatusBadge status={r.status} />
                  </td>
                  <td className="p-3 text-slate-200">
                    {r.target_column || <span className="text-amber-400 italic">None (NEEDS_INPUT)</span>}
                  </td>
                  <td className="p-3 text-slate-300 uppercase">
                    {r.target_metric || "Auto"}
                  </td>
                  <td className="p-3 text-slate-300">{r.dataset_version}</td>
                  <td className="p-3">
                    {r.best_metric_value !== undefined && r.best_metric_value !== null ? (
                      <span className="font-bold text-emerald-400">{r.best_metric_value.toFixed(4)}</span>
                    ) : (
                      <span className="text-slate-500">—</span>
                    )}
                  </td>
                  <td className="p-3 text-slate-400 text-[11px]">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                  <td className="p-3 text-right">
                    <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-slate-300 transition" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
