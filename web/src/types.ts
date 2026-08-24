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
  latency_ms: { camera?: number; decode: number; perception: number; risk_control: number; actuation?: number; thermal_penalty?: number; render: number; total: number };
  command: { speed: number; yaw_rate: number; brake: number };
  virtual_hardware?: {
    profile: string; simulated: boolean;
    camera: { width: number; height: number; target_fps: number };
    declared_cpu_cores: number; declared_memory_mb: number;
    temperature_c: number; throttled: boolean; gpu_available: boolean;
    fallback_active: boolean; frame_reused: boolean; deadline_ms: number;
    applied_command: { speed: number; yaw_rate: number; brake: number };
  };
  evaluation_only: { nearest_obstacle_m: number | null; true_ttc_s: number | null; collision: boolean };
  preview_jpeg?: string;
};

declare global { interface Window { FLOWGUARD_REPLAY?: Telemetry[] } }
