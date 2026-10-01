export interface Project {
  id: string;
  name: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

export interface Dataset {
  id: string;
  project_id: string;
  version: string;
  filename: string;
  file_path: string;
  row_count: number;
  col_count: number;
  created_at: string;
}

export interface WorkflowRun {
  id: string;
  project_id: string;
  dataset_version: string;
  target_column?: string;
  target_metric?: string;
  status: "PENDING" | "RUNNING" | "NEEDS_INPUT" | "SUCCESS" | "FAILED" | "CANCELLED";
  current_stage: string;
  iteration: number;
  best_experiment_id?: string;
  best_metric_value?: number;
  error?: string;
  degraded?: boolean;
  created_at: string;
  completed_at?: string;
}

export interface ArtifactIndex {
  id: string;
  project_id: string;
  stage?: string;
  artifact_type: string;
  path?: string;
  file_path?: string;
  version?: string;
  summary?: string;
  created_at: string;
}

export interface ArtifactContent {
  type: "json" | "markdown" | "code" | "table" | "text";
  data?: any;
  content?: string;
  raw?: string;
  language?: string;
  columns?: string[];
  rows?: Record<string, any>[];
  total_rows?: number;
  total_columns?: number;
}

export interface DatasetPreview {
  version: string;
  filename: string;
  columns: string[];
  rows: Record<string, any>[];
  total_rows: number;
  total_columns: number;
}

export type EventType =
  | "WORKFLOW_STARTED"
  | "AGENT_STARTED"
  | "AGENT_COMPLETED"
  | "TOOL_STARTED"
  | "TOOL_COMPLETED"
  | "CODE_EXECUTION_STARTED"
  | "CODE_EXECUTION_COMPLETED"
  | "ARTIFACT_CREATED"
  | "EXPERIMENT_STARTED"
  | "EXPERIMENT_COMPLETED"
  | "EVALUATION_FAILED"
  | "SUPERVISOR_DECISION"
  | "supervisor_decision"
  | "worker_report"
  | "guard_override"
  | "concern"
  | "WORKFLOW_COMPLETED"
  | (string & {});

export interface PipelineEvent {
  id: string;
  project_id: string;
  run_id: string;
  event_type: EventType;
  stage: string;
  message: string;
  data: Record<string, any>;
  timestamp: string;
}

export interface LeaderboardItem {
  run_id: string;
  experiment_id: string;
  model_name?: string;
  model_type?: string;
  target_metric: string;
  score?: number | null;
  metric_value?: number | null;
  dataset_version: string;
  feature_version?: string;
  duration_seconds?: number;
  status?: string;
  params: Record<string, any>;
  metrics: Record<string, any>;
  tags: Record<string, any>;
  artifact_uri?: string;
}

export interface LeaderboardResponse {
  project_id: string;
  target_metric: string;
  direction: "max" | "min";
  leaderboard: LeaderboardItem[];
  error?: string;
}

export interface EvaluationIssue {
  type: string;
  severity: "HIGH" | "MEDIUM" | "LOW" | "CRITICAL";
  evidence: string;
  implication?: string;
}

export interface EvaluationResponse {
  json: {
    verdict?: "PASS" | "IMPROVE" | "STOP";
    issues?: EvaluationIssue[];
    recommendations?: string[];
    metrics?: Record<string, any>;
    iteration?: number;
    best_candidate?: string;
    decision?: string;
    [key: string]: any;
  };
  markdown: string;
}

export interface ReportResponse {
  summary: {
    model_name?: string;
    best_score?: number;
    target_metric?: string;
    dataset_version?: string;
    feature_version?: string;
    iterations_run?: number;
    verdict?: string;
    [key: string]: any;
  };
  markdown: string;
}

export interface CodeExecution {
  id: string;
  project_id: string;
  run_id?: string;
  stage: string;
  script_name: string;
  task_description?: string;
  attempt?: number;
  code: string;
  exit_code: number;
  stdout: string;
  stderr: string;
  success: boolean;
  executed_at: string;
  duration_ms?: number;
}

