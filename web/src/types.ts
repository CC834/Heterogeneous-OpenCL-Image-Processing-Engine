export type Telemetry = {
  schema_version: number;
  frame_id: number;
  simulation_time_s: number;
  mode: string;
  fps: number;
  deadline_missed: boolean;
  gpu_ratio: number;
  sector_risk: [number, number, number];
  minimum_ttc_s: number | null;
  warning: "clear" | "yellow" | "red";
  latency_ms: { decode: number; perception: number; risk_control: number; render: number; total: number };
  command: { speed: number; yaw_rate: number; brake: number };
  evaluation_only: { nearest_obstacle_m: number | null; true_ttc_s: number | null; collision: boolean };
  preview_jpeg?: string;
};

declare global { interface Window { FLOWGUARD_REPLAY?: Telemetry[] } }
