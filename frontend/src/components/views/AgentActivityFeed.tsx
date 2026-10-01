import { useState, useRef, useEffect, useMemo } from "react";
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Terminal,
  AlertCircle,
  Cpu,
  CheckCircle2,
  Clock,
  ShieldAlert,
  ArrowRight,
  Table,
  ListTree,
  ExternalLink,
  Target,
} from "lucide-react";
import { cn as clsx } from "../../utils/cn";
import { PipelineEvent, WorkflowRun } from "../../types";
import { Badge, StatusBadge } from "../common/Badge";

interface AgentActivityFeedProps {
  events: PipelineEvent[];
  activeRun?: WorkflowRun;
  isConnected: boolean;
  isCompleted: boolean;
  error?: string | null;
  onSelectStage?: (stage: string) => void;
}

interface StepGroup {
  step: number;
  supervisorEvent?: PipelineEvent;
  guardOverrides: PipelineEvent[];
  workerReports: PipelineEvent[];
  concerns: PipelineEvent[];
  otherEvents: PipelineEvent[];
}

export function AgentActivityFeed({
  events,
  activeRun,
  isConnected,
  isCompleted: _isCompleted,
  error,
  onSelectStage,
}: AgentActivityFeedProps) {
  const [activeView, setActiveView] = useState<"timeline" | "ledger" | "raw">("timeline");
  const [expandedItems, setExpandedItems] = useState<Record<string, boolean>>({});
  const feedEndRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll on new events if connected
  useEffect(() => {
    if (isConnected && feedEndRef.current) {
      feedEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [events.length, isConnected]);

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  // Group events by pipeline turn / step
  const stepGroups = useMemo(() => {
    const groups: StepGroup[] = [];
    let currentStep = 0;
    let currentGroup: StepGroup = {
      step: 0,
      guardOverrides: [],
      workerReports: [],
      concerns: [],
      otherEvents: [],
    };

    events.forEach((ev) => {
      const et = ev.event_type;
      const data = ev.data || {};

      if (et === "supervisor_decision" || et === "SUPERVISOR_DECISION") {
        if (data.step !== undefined && data.step !== currentStep && currentGroup.supervisorEvent) {
          groups.push(currentGroup);
          currentStep = Number(data.step);
          currentGroup = {
            step: currentStep,
            guardOverrides: [],
            workerReports: [],
            concerns: [],
            otherEvents: [],
          };
        }
        currentGroup.supervisorEvent = ev;
        if (data.step !== undefined) {
          currentGroup.step = Number(data.step);
        }
      } else if (et === "guard_override") {
        currentGroup.guardOverrides.push(ev);
      } else if (et === "worker_report" || et === "AGENT_COMPLETED") {
        if (et === "worker_report") {
          currentGroup.workerReports.push(ev);
        }
      } else if (et === "concern") {
        currentGroup.concerns.push(ev);
      } else {
        currentGroup.otherEvents.push(ev);
      }
    });

    if (currentGroup.supervisorEvent || currentGroup.workerReports.length > 0 || currentGroup.otherEvents.length > 0) {
      groups.push(currentGroup);
    }

    return groups;
  }, [events]);

  // Build rows for the Ledger Table view
  const ledgerRows = useMemo(() => {
    const rows: Array<{
      id: string;
      step: number;
      time: string;
      agent: string;
      action: string;
      objective: string;
      status: string;
      score: number | null;
      reason: string;
    }> = [];

    stepGroups.forEach((sg) => {
      const sData = sg.supervisorEvent?.data || {};
      const action = sData.action || sData.next_action || "unknown";
      const reason = sData.reason || "";
      const objective = sData.brief_summary || sData.brief?.objective || "";

      if (sg.workerReports.length > 0) {
        sg.workerReports.forEach((wr) => {
          const wData = wr.data || {};
          rows.push({
            id: wr.id,
            step: sg.step,
            time: wr.timestamp,
            agent: wData.agent || wr.stage,
            action,
            objective,
            status: wData.status || "ok",
            score: wData.score !== undefined ? wData.score : null,
            reason,
          });
        });
      } else if (sg.supervisorEvent) {
        rows.push({
          id: sg.supervisorEvent.id,
          step: sg.step,
          time: sg.supervisorEvent.timestamp,
          agent: "supervisor",
          action,
          objective,
          status: "decided",
          score: null,
          reason,
        });
      }
    });

    return rows;
  }, [stepGroups]);

  const mapActionToStage = (action: string) => {
    switch (action.toLowerCase()) {
      case "profile":
        return "profile";
      case "eda":
        return "eda";
      case "fe":
      case "features":
        return "features";
      case "model":
      case "modeling":
        return "models";
      case "judge":
      case "evaluator":
        return "evaluation";
      case "report":
        return "report";
      default:
        return "overview";
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden bg-[#0a0f1d]">
      {/* Top Banner: Supervisor Control & Status */}
      <div className="p-4 border-b border-slate-800/80 bg-slate-900/40 select-none">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-purple-500/10 border border-purple-500/30 flex items-center justify-center text-purple-400 shadow-sm shadow-purple-500/20">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-semibold text-slate-100 uppercase font-mono tracking-wider flex items-center gap-2">
                Supervisor-Led Autonomous Team
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 font-sans border border-purple-500/30">
                  V2 Pipeline
                </span>
              </div>
              <div className="text-[11px] text-slate-400 font-sans">
                Real-time agent deliberations, deterministic guards, and worker reports
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {activeRun && (
              <div className="flex items-center gap-2 text-xs font-mono text-slate-400 bg-slate-900/80 px-2.5 py-1 rounded-md border border-slate-800">
                <span>Version:</span>
                <span className="text-sky-400 font-bold">{activeRun.iteration ? `v${activeRun.iteration}` : "v1"}</span>
                {activeRun.best_metric_value !== undefined && activeRun.best_metric_value !== null && (
                  <>
                    <span className="text-slate-600">|</span>
                    <span>Best Score:</span>
                    <span className="text-emerald-400 font-bold">{Number(activeRun.best_metric_value).toFixed(4)}</span>
                  </>
                )}
              </div>
            )}
            {activeRun && <StatusBadge status={activeRun.status} />}
            {isConnected && (
              <span className="flex items-center gap-1.5 text-[11px] font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
                LIVE
              </span>
            )}
          </div>
        </div>

        {/* View Switcher Controls */}
        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-1.5 bg-slate-950/60 p-1 rounded-lg border border-slate-800/80">
            <button
              onClick={() => setActiveView("timeline")}
              className={clsx(
                "flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono transition",
                activeView === "timeline"
                  ? "bg-purple-600 text-white shadow-sm shadow-purple-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <ListTree className="w-3.5 h-3.5" />
              Step Timeline
            </button>
            <button
              onClick={() => setActiveView("ledger")}
              className={clsx(
                "flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono transition",
                activeView === "ledger"
                  ? "bg-purple-600 text-white shadow-sm shadow-purple-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Table className="w-3.5 h-3.5" />
              Ledger Table ({ledgerRows.length})
            </button>
            <button
              onClick={() => setActiveView("raw")}
              className={clsx(
                "flex items-center gap-1.5 px-3 py-1 rounded-md text-xs font-mono transition",
                activeView === "raw"
                  ? "bg-purple-600 text-white shadow-sm shadow-purple-600/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              )}
            >
              <Terminal className="w-3.5 h-3.5" />
              Raw Stream ({events.length})
            </button>
          </div>

          <div className="text-[11px] font-mono text-slate-500">
            Steps Recorded: <span className="text-purple-300 font-semibold">{stepGroups.length}</span>
          </div>
        </div>
      </div>

      {/* Main View Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/40 rounded-lg flex items-center gap-2.5 text-xs font-mono text-rose-300 shadow-sm">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <div className="flex-1">
              <span className="font-semibold text-rose-200">Execution Alert:</span> {error}
            </div>
          </div>
        )}

        {activeRun?.status === "FAILED" && !error && (
          <div className="p-3 bg-rose-500/10 border border-rose-500/40 rounded-lg flex items-center gap-2.5 text-xs font-mono text-rose-300 shadow-sm">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <div className="flex-1">
              <span className="font-semibold text-rose-200">Run Terminated:</span>{" "}
              {activeRun.error || "The autonomous workflow failed."}
            </div>
          </div>
        )}

        {events.length === 0 && (
          <div className="h-64 flex flex-col items-center justify-center text-slate-500 font-mono text-xs gap-3">
            <div className="w-12 h-12 rounded-xl bg-purple-500/5 border border-purple-500/20 flex items-center justify-center text-purple-400/60 animate-pulse">
              <Cpu className="w-6 h-6" />
            </div>
            <span>Supervisor initializing autonomous run loop...</span>
          </div>
        )}

        {/* 1. TIMELINE VIEW */}
        {activeView === "timeline" && (
          <div className="space-y-4">
            {stepGroups.map((sg) => {
              const sData = sg.supervisorEvent?.data || {};
              const action = sData.action || sData.next_action || "evaluating";
              const brief = sData.brief || {};
              const objective = sData.brief_summary || brief.objective || "";
              const reason = sData.reason || "";
              const thought = sData.thought || "";
              const isSupervisorExpanded = expandedItems[`sup_${sg.step}`];

              return (
                <div
                  key={`step_${sg.step}`}
                  className="rounded-xl border border-slate-800/80 bg-slate-900/40 shadow-sm overflow-hidden"
                >
                  {/* Step Header */}
                  <div className="px-4 py-2.5 bg-slate-900/90 border-b border-slate-800/80 flex items-center justify-between">
                    <div className="flex items-center gap-2.5">
                      <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40">
                        STEP #{sg.step}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-slate-500" />
                      <span className="text-xs font-mono font-bold uppercase tracking-wider text-sky-400">
                        Target: {action}
                      </span>
                      {brief.mode && (
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                          mode={brief.mode}
                        </span>
                      )}
                    </div>

                    <div className="text-[11px] font-mono text-slate-500 flex items-center gap-1.5">
                      <Clock className="w-3 h-3" />
                      {sg.supervisorEvent ? new Date(sg.supervisorEvent.timestamp).toLocaleTimeString() : ""}
                    </div>
                  </div>

                  <div className="p-4 space-y-3">
                    {/* Guard Override Banner if any */}
                    {sg.guardOverrides.map((go) => (
                      <div
                        key={go.id}
                        className="p-2.5 rounded-lg bg-rose-950/30 border border-rose-800/40 text-rose-300 text-xs font-mono flex items-start gap-2"
                      >
                        <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-bold text-rose-200">Guard Intervention:</span>{" "}
                          {go.data?.reason || go.message}
                        </div>
                      </div>
                    ))}

                    {/* Supervisor Decision Block */}
                    <div className="p-3.5 rounded-lg bg-purple-950/20 border border-purple-800/30 space-y-2">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Cpu className="w-4 h-4 text-purple-400" />
                          <span className="text-xs font-mono font-bold text-purple-200 uppercase">
                            Supervisor Deliberation
                          </span>
                        </div>
                        <button
                          onClick={() => toggleExpand(`sup_${sg.step}`)}
                          className="text-[11px] font-mono text-purple-400 hover:text-purple-300 flex items-center gap-1"
                        >
                          {isSupervisorExpanded ? "Hide Details" : "View Brief & Context"}
                          {isSupervisorExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        </button>
                      </div>

                      {reason && (
                        <div className="text-xs font-sans text-slate-200 leading-relaxed">
                          <span className="text-purple-300 font-semibold font-mono text-[11px]">Reason:</span> {reason}
                        </div>
                      )}

                      {objective && (
                        <div className="text-xs font-sans text-slate-300 flex items-start gap-1.5 pt-0.5">
                          <Target className="w-3.5 h-3.5 text-sky-400 shrink-0 mt-0.5" />
                          <span>
                            <strong className="text-slate-200">Brief:</strong> {objective}
                          </span>
                        </div>
                      )}

                      {isSupervisorExpanded && (
                        <div className="pt-2 border-t border-purple-800/40 space-y-2 text-xs font-mono">
                          {thought && (
                            <div className="p-2 rounded bg-black/40 text-slate-400 text-[11px]">
                              <span className="text-purple-300 font-semibold">Thought:</span> {thought}
                            </div>
                          )}
                          {brief.focus_points && brief.focus_points.length > 0 && (
                            <div>
                              <span className="text-slate-400 text-[10px] uppercase font-semibold">Focus Points:</span>
                              <ul className="list-disc list-inside text-slate-300 text-[11px] pl-1 pt-0.5 space-y-0.5">
                                {brief.focus_points.map((fp: string, idx: number) => (
                                  <li key={idx}>{fp}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Specialist Worker Reports */}
                    {sg.workerReports.map((wr) => {
                      const wData = wr.data || {};
                      const agentName = wData.agent || wr.stage;
                      const isOk = wData.status === "ok";
                      const evidence = wData.evidence || {};
                      const isWrExpanded = expandedItems[`wr_${wr.id}`];

                      return (
                        <div
                          key={wr.id}
                          className={clsx(
                            "p-3.5 rounded-lg border text-xs font-mono space-y-2.5 transition",
                            isOk
                              ? "bg-slate-900/70 border-slate-800 hover:border-slate-700"
                              : "bg-rose-950/20 border-rose-800/40"
                          )}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {isOk ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                              ) : (
                                <AlertTriangle className="w-4 h-4 text-rose-400" />
                              )}
                              <span className="font-bold text-slate-100 uppercase tracking-wide">
                                Specialist: {agentName}
                              </span>
                              <Badge variant={isOk ? "success" : "error"}>{wData.status || "OK"}</Badge>
                            </div>

                            <div className="flex items-center gap-2">
                              {onSelectStage && (
                                <button
                                  onClick={() => onSelectStage(mapActionToStage(agentName))}
                                  className="text-[11px] text-sky-400 hover:text-sky-300 flex items-center gap-1 bg-sky-500/10 px-2 py-0.5 rounded border border-sky-500/20 transition"
                                >
                                  Open {agentName.toUpperCase()}
                                  <ExternalLink className="w-3 h-3" />
                                </button>
                              )}
                              <button
                                onClick={() => toggleExpand(`wr_${wr.id}`)}
                                className="text-slate-500 hover:text-slate-300"
                              >
                                {isWrExpanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                              </button>
                            </div>
                          </div>

                          <div className="text-slate-200 font-sans text-xs leading-relaxed">
                            {wData.result_summary || wr.message}
                          </div>

                          {/* Evidence Chips */}
                          <div className="flex items-center gap-2 flex-wrap pt-1">
                            {wData.score !== undefined && wData.score !== null && (
                              <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 font-mono text-[11px]">
                                Metric Score: <strong>{Number(wData.score).toFixed(4)}</strong>
                              </span>
                            )}
                            {evidence.best_model && (
                              <span className="px-2 py-0.5 rounded bg-sky-500/10 border border-sky-500/30 text-sky-300 font-mono text-[11px]">
                                Best Candidate: <strong>{evidence.best_model}</strong>
                              </span>
                            )}
                            {evidence.feature_count !== undefined && (
                              <span className="px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/30 text-blue-300 font-mono text-[11px]">
                                Features Created: <strong>{evidence.feature_count}</strong>
                              </span>
                            )}
                            {evidence.overall && (
                              <span className="px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/30 text-purple-300 font-mono text-[11px]">
                                Judge Verdict: <strong>{evidence.overall.toUpperCase()}</strong>
                              </span>
                            )}
                          </div>

                          {/* Expanded Raw Payload */}
                          {isWrExpanded && (
                            <div className="pt-2 border-t border-slate-800 text-[11px]">
                              <pre className="p-2.5 rounded bg-black/40 border border-slate-800 text-sky-300 overflow-x-auto">
                                {JSON.stringify(wr.data, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      );
                    })}

                    {/* Worker Concerns Raised */}
                    {sg.concerns.map((cn) => {
                      const cData = cn.data || {};
                      return (
                        <div
                          key={cn.id}
                          className="p-3 rounded-lg bg-amber-950/20 border border-amber-800/40 text-xs font-mono space-y-1.5"
                        >
                          <div className="flex items-center gap-2 text-amber-300 font-semibold">
                            <AlertCircle className="w-4 h-4 text-amber-400" />
                            <span>Worker Concern Raised ({cData.agent}):</span>
                          </div>
                          <div className="text-slate-200 font-sans text-xs">
                            <span className="font-semibold text-amber-200">Claim:</span> {cData.claim}
                          </div>
                          {cData.evidence && (
                            <div className="text-slate-300 text-[11px]">
                              <span className="font-semibold text-amber-300 font-mono">Evidence:</span> {cData.evidence}
                            </div>
                          )}
                          {cData.suggestion && (
                            <div className="text-slate-300 text-[11px]">
                              <span className="font-semibold text-emerald-400 font-mono">Suggestion:</span>{" "}
                              {cData.suggestion}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* 2. RUN LEDGER TABLE VIEW */}
        {activeView === "ledger" && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs border-collapse">
                <thead>
                  <tr className="bg-slate-900/90 text-slate-400 border-b border-slate-800 select-none">
                    <th className="p-3 font-semibold">Step</th>
                    <th className="p-3 font-semibold">Time</th>
                    <th className="p-3 font-semibold">Agent</th>
                    <th className="p-3 font-semibold">Action</th>
                    <th className="p-3 font-semibold">Status</th>
                    <th className="p-3 font-semibold">Score</th>
                    <th className="p-3 font-semibold">Objective / Brief</th>
                    <th className="p-3 font-semibold">Decision Reason</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-300">
                  {ledgerRows.map((row) => (
                    <tr key={row.id} className="hover:bg-slate-800/30 transition">
                      <td className="p-3 font-bold text-purple-400">#{row.step}</td>
                      <td className="p-3 text-[11px] text-slate-500 whitespace-nowrap">
                        {new Date(row.time).toLocaleTimeString()}
                      </td>
                      <td className="p-3 uppercase font-semibold text-sky-400">{row.agent}</td>
                      <td className="p-3 font-semibold text-slate-200">{row.action}</td>
                      <td className="p-3">
                        <Badge variant={row.status === "ok" ? "success" : row.status === "failed" ? "error" : "purple"}>
                          {row.status}
                        </Badge>
                      </td>
                      <td className="p-3 font-bold text-emerald-400">
                        {row.score !== null ? Number(row.score).toFixed(4) : "-"}
                      </td>
                      <td className="p-3 text-slate-300 font-sans max-w-xs truncate" title={row.objective}>
                        {row.objective || "-"}
                      </td>
                      <td className="p-3 text-slate-400 font-sans max-w-sm truncate" title={row.reason}>
                        {row.reason || "-"}
                      </td>
                    </tr>
                  ))}
                  {ledgerRows.length === 0 && (
                    <tr>
                      <td colSpan={8} className="p-6 text-center text-slate-500">
                        No ledger rows recorded yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* 3. RAW EVENT LOG VIEW */}
        {activeView === "raw" && (
          <div className="space-y-2">
            {events.map((ev) => {
              const isExpanded = expandedItems[ev.id];
              const hasData = ev.data && Object.keys(ev.data).length > 0;

              return (
                <div key={ev.id} className="rounded-lg border border-slate-800/80 bg-slate-900/60 text-xs font-mono">
                  <div
                    onClick={() => hasData && toggleExpand(ev.id)}
                    className="p-3 flex items-start justify-between gap-3 cursor-pointer select-none"
                  >
                    <div className="flex items-start gap-2.5 min-w-0">
                      <div className="min-w-0 space-y-1">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="uppercase text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-800 text-slate-300">
                            {ev.stage}
                          </span>
                          <span className="text-[10px] text-purple-400 font-bold">{ev.event_type}</span>
                          <span className="text-[10px] text-slate-500">
                            {new Date(ev.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <div className="text-slate-200 font-sans text-xs">{ev.message}</div>
                      </div>
                    </div>
                    {hasData && (
                      <button className="text-slate-500 mt-1">
                        {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                      </button>
                    )}
                  </div>
                  {isExpanded && hasData && (
                    <div className="px-3 pb-3 border-t border-slate-800/60 bg-black/40 rounded-b-lg">
                      <pre className="p-2 rounded text-sky-300 text-[11px] overflow-x-auto">
                        {JSON.stringify(ev.data, null, 2)}
                      </pre>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div ref={feedEndRef} />
      </div>
    </div>
  );
}
