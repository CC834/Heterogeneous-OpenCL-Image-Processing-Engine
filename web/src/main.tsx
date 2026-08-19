import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import type { Telemetry } from "./types";
import "./style.css";
import "./controls.css";

const empty: Telemetry = {
  schema_version: 2, frame_id: 0, simulation_time_s: 0, mode: "waiting", fps: 0,
  deadline_missed: false, gpu_ratio: 0, sector_risk: [0, 0, 0], minimum_ttc_s: null,
  warning: "clear", latency_ms: { camera: 0, decode: 0, perception: 0, risk_control: 0, actuation: 0, thermal_penalty: 0, render: 0, total: 0 },
  command: { speed: 0, yaw_rate: 0, brake: 0 },
  virtual_hardware: { profile: "desktop-native", simulated: false,
    camera: { width: 640, height: 360, target_fps: 30 }, declared_cpu_cores: 0,
    declared_memory_mb: 0, temperature_c: 0, throttled: false, gpu_available: true,
    fallback_active: false, frame_reused: false, deadline_ms: 33.333,
    applied_command: { speed: 0, yaw_rate: 0, brake: 0 } },
  evaluation_only: { nearest_obstacle_m: null, true_ttc_s: null, collision: false }
};

function Sparkline({ values }: { values: number[] }) {
  const points = values.map((v, i) => `${i * 300 / Math.max(1, values.length - 1)},${80 - Math.min(75, v)}`).join(" ");
  return <svg viewBox="0 0 300 80" aria-label="Latency history"><polyline points={points} /></svg>;
}

function App() {
  const replay = window.FLOWGUARD_REPLAY;
  const [index, setIndex] = useState(0);
  const [live, setLive] = useState<Telemetry>(replay?.[0] ?? empty);
  const [history, setHistory] = useState<number[]>([]);
  const [socket, setSocket] = useState<WebSocket>();
  useEffect(() => {
    if (replay) return;
    const socket = new WebSocket(`ws://${location.host}/ws`);
    setSocket(socket);
    socket.onmessage = event => {
      const next = JSON.parse(event.data) as Telemetry;
      if (![1, 2].includes(next.schema_version) || next.frame_id === undefined) return;
      setLive(next); setHistory(values => [...values.slice(-89), next.latency_ms.total]);
    };
    return () => socket.close();
  }, [replay]);
  const telemetry = replay?.[index] ?? live;
  const hardware = telemetry.virtual_hardware ?? empty.virtual_hardware!;
  const allocation = Math.round(telemetry.gpu_ratio * 100);
  const risk = useMemo(() => telemetry.sector_risk.map(value => Math.round(value * 100)), [telemetry]);
  const send = (action: string, value?: string) => socket?.send(JSON.stringify({ action, value }));
  return <main>
    <header><div><span className="eyebrow">VISUAL COLLISION AWARENESS · ONBOARD COMPUTE STUDY</span><h1>FlowGuard <em>OpenCL</em></h1></div><div className="controls"><button onClick={()=>send("pause")}>Pause</button><button onClick={()=>send("reset")}>Reset</button><select defaultValue={telemetry.mode} onChange={e=>send("mode",e.target.value)}><option>cpu</option><option>gpu</option><option>fixed</option><option>adaptive</option></select><div className={`status ${telemetry.warning}`}>{telemetry.warning}</div></div></header>
    <section className="hero">
      <div className="camera">{telemetry.preview_jpeg ? <img src={`data:image/jpeg;base64,${telemetry.preview_jpeg}`}/> : replay ? <video src="annotated.mp4" controls/> : <div className="waiting">Waiting for 10 FPS preview</div>}<div className="reticle">＋</div></div>
      <aside><div className="metric"><label>TTC proxy</label><strong>{telemetry.minimum_ttc_s?.toFixed(2) ?? "—"}<small>s</small></strong></div><div className="metric"><label>Pipeline</label><strong>{telemetry.latency_ms.total.toFixed(1)}<small>ms</small></strong></div><div className="metric"><label>Throughput</label><strong>{telemetry.fps.toFixed(1)}<small>fps</small></strong></div></aside>
    </section>
    <section className="grid">
      <article><h2>Collision sectors</h2><div className="sectors">{["LEFT","CENTRE","RIGHT"].map((name,i)=><div key={name}><span>{name}</span><i style={{height:`${risk[i]}%`}}/><b>{risk[i]}%</b></div>)}</div></article>
      <article><h2>Heterogeneous allocation</h2><div className="allocation"><span style={{width:`${100-allocation}%`}}>CPU {100-allocation}%</span><span style={{width:`${allocation}%`}}>GPU {allocation}%</span></div><p>{telemetry.mode} scheduler · frame {telemetry.frame_id}</p></article>
      <article><h2>Latency trace</h2><Sparkline values={replay ? replay.slice(0,index+1).map(t=>t.latency_ms.total) : history}/></article>
      <article><h2>Avoidance command</h2><dl><div><dt>Speed</dt><dd>{telemetry.command.speed.toFixed(2)} m/s</dd></div><div><dt>Yaw rate</dt><dd>{telemetry.command.yaw_rate.toFixed(1)}°/s</dd></div><div><dt>Brake</dt><dd>{Math.round(telemetry.command.brake*100)}%</dd></div></dl></article>
      <article className="hardware"><h2>Virtual onboard hardware</h2><div className="hardware-title"><strong>{hardware.profile}</strong><span className={hardware.simulated ? "sim" : "measured"}>{hardware.simulated ? "SIMULATED" : "HOST"}</span></div><dl><div><dt>Camera</dt><dd>{hardware.camera.width}×{hardware.camera.height} · {hardware.camera.target_fps} Hz</dd></div><div><dt>Temperature</dt><dd>{hardware.simulated ? `${hardware.temperature_c.toFixed(0)}°C` : "not modelled"}</dd></div><div><dt>GPU</dt><dd>{hardware.gpu_available ? "available" : "unavailable"}{hardware.fallback_active ? " · CPU failover" : ""}</dd></div><div><dt>Camera / actuator</dt><dd>{(telemetry.latency_ms.camera ?? 0).toFixed(1)} / {(telemetry.latency_ms.actuation ?? 0).toFixed(1)} ms</dd></div></dl>{hardware.simulated && <p>Declared constraint model—never reported as physical-board evidence.</p>}</article>
    </section>
    {replay && <footer><button onClick={()=>setIndex(0)}>Reset</button><input type="range" min="0" max={replay.length-1} value={index} onChange={e=>setIndex(Number(e.target.value))}/><span>{index+1}/{replay.length}</span></footer>}
  </main>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App/></React.StrictMode>);
