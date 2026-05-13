import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  CheckCircle2,
  Download,
  Languages,
  Loader2,
  Play,
  Search,
  ShieldCheck,
} from "lucide-react";
import { fetchHealth, fetchTopics, streamGeneration } from "./api.js";

const COPY = {
  en: {
    title: "Adversarial Debate Argument Generator",
    subtitle: "Single-pass generation compared with an internal four-agent debate.",
    topic: "Motion",
    custom: "Custom motion",
    side: "Target side",
    pro: "Pro",
    con: "Con",
    start: "Generate",
    running: "Generating",
    healthReady: "Ready",
    healthNoSearch: "Search key missing",
    single: "Single Agent",
    adversarial: "Adversarial Debate",
    final: "Final",
    timeline: "Agent Timeline",
    sources: "Sources",
    metrics: "Metrics",
    export: "Export JSON",
    useCustom: "Use custom",
    useSearch: "Live search",
    useCache: "Cache",
    empty: "Run a debate to compare both strategies.",
    streamingHint: "Tokens will appear here as each agent speaks.",
  },
  zh: {
    title: "对抗式辩论论点生成系统",
    subtitle: "单次生成与四 Agent 内部对抗的并排比较。",
    topic: "辩题",
    custom: "自定义辩题",
    side: "目标立场",
    pro: "正方",
    con: "反方",
    start: "开始生成",
    running: "生成中",
    healthReady: "就绪",
    healthNoSearch: "缺少搜索密钥",
    single: "单 Agent",
    adversarial: "对抗式辩论",
    final: "最终结果",
    timeline: "Agent 时间线",
    sources: "来源",
    metrics: "指标",
    export: "导出 JSON",
    useCustom: "使用自定义",
    useSearch: "实时搜索",
    useCache: "缓存",
    empty: "运行一次辩论后对比两种策略。",
    streamingHint: "每个 Agent 发言时，token 会实时出现在这里。",
  },
};

export default function App() {
  const [language, setLanguage] = useState("en");
  const [topics, setTopics] = useState([]);
  const [topicId, setTopicId] = useState("");
  const [customTopic, setCustomTopic] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [targetSide, setTargetSide] = useState("pro");
  const [useSearch, setUseSearch] = useState(true);
  const [useCache, setUseCache] = useState(true);
  const [health, setHealth] = useState(null);
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState([]);
  const [agents, setAgents] = useState([]);
  const [singleSegments, setSingleSegments] = useState([]);
  const [adversarialSegments, setAdversarialSegments] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const t = COPY[language];

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth({ app: "offline" }));
  }, []);

  useEffect(() => {
    fetchTopics(language).then((items) => {
      setTopics(items);
      setTopicId((current) => current || items[0]?.id || "");
    });
  }, [language]);

  const selectedTopic = useMemo(() => topics.find((item) => item.id === topicId), [topics, topicId]);
  const activeTopic = useCustom ? customTopic : selectedTopic?.label || "";

  async function run() {
    if (!activeTopic.trim()) return;
    setRunning(true);
    setStages([]);
    setAgents([]);
    setSingleSegments([]);
    setAdversarialSegments([]);
    setResult(null);
    setError("");
    try {
      await streamGeneration(
        {
          topic: activeTopic.trim(),
          target_side: targetSide,
          language,
          use_search: useSearch,
          use_cache: useCache,
        },
        {
          stage: (payload) => setStages((items) => [...items, payload]),
          agent_start: (payload) => startSegment(payload),
          token: (payload) => appendToken(payload),
          agent_done: (payload) => finishSegment(payload),
          panel_append: (payload) => appendCachedSegment(payload),
          agent: (payload) => setAgents((items) => [...items, payload]),
          final: (payload) => setResult(payload),
          error: (payload) => setError(payload.message || "Generation failed"),
          done: () => setRunning(false),
        },
      );
    } catch (err) {
      setError(err.message);
      setRunning(false);
    }
  }

  function updatePanel(panel, updater) {
    const setter = panel === "single" ? setSingleSegments : setAdversarialSegments;
    setter((items) => updater(items));
  }

  function startSegment(payload) {
    updatePanel(payload.panel, (items) => {
      const key = segmentKey(payload);
      if (items.some((item) => item.key === key)) return items;
      return [
        ...items,
        {
          key,
          stage: payload.stage,
          role: payload.role,
          side: payload.side,
          content: "",
          status: "running",
        },
      ];
    });
  }

  function appendToken(payload) {
    updatePanel(payload.panel, (items) => {
      const key = segmentKey(payload);
      const index = items.findIndex((item) => item.key === key);
      if (index === -1) {
        return [
          ...items,
          {
            key,
            stage: payload.stage,
            role: payload.role,
            content: payload.token,
            status: "running",
          },
        ];
      }
      return items.map((item, itemIndex) =>
        itemIndex === index ? { ...item, content: item.content + payload.token } : item,
      );
    });
  }

  function finishSegment(payload) {
    updatePanel(payload.panel, (items) =>
      items.map((item) =>
        item.key === segmentKey(payload)
          ? { ...item, content: payload.content || item.content, status: "done" }
          : item,
      ),
    );
  }

  function appendCachedSegment(payload) {
    updatePanel(payload.panel, (items) => [
      ...items,
      {
        key: `${payload.panel}-${payload.stage}-${payload.role}-${items.length}`,
        stage: payload.stage,
        role: payload.role,
        content: payload.content,
        status: "done",
      },
    ]);
  }

  function exportJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `debate-result-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <main className="min-h-screen bg-[oklch(0.965_0.009_215)] text-[oklch(0.18_0.018_225)]">
      <section className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 lg:px-8">
        <header className="topbar">
          <div>
            <p className="eyebrow">DTS407TC A2</p>
            <h1>{t.title}</h1>
            <p className="subtitle">{t.subtitle}</p>
          </div>
          <div className="status-strip">
            <span className={health?.tavily_configured ? "status good" : "status warn"}>
              <ShieldCheck size={16} />
              {health?.tavily_configured ? t.healthReady : t.healthNoSearch}
            </span>
            <button
              className="icon-button"
              onClick={() => setLanguage(language === "en" ? "zh" : "en")}
              title="Language"
              aria-label={language === "en" ? "Switch to Chinese" : "Switch to English"}
            >
              <Languages size={18} />
              {language.toUpperCase()}
            </button>
          </div>
        </header>

        <section className="control-surface">
          <label>
            <span>{t.topic}</span>
            <select value={topicId} onChange={(event) => setTopicId(event.target.value)} disabled={useCustom}>
              {topics.map((topic) => (
                <option key={topic.id} value={topic.id}>
                  {topic.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span>{t.custom}</span>
            <input value={customTopic} onChange={(event) => setCustomTopic(event.target.value)} placeholder={t.custom} />
          </label>
          <Segmented
            label={t.side}
            value={targetSide}
            onChange={setTargetSide}
            options={[
              { value: "pro", label: t.pro },
              { value: "con", label: t.con },
            ]}
          />
          <div className="toggle-row">
            <Toggle label={t.useCustom} checked={useCustom} onChange={setUseCustom} />
            <Toggle label={t.useSearch} checked={useSearch} onChange={setUseSearch} />
            <Toggle label={t.useCache} checked={useCache} onChange={setUseCache} />
          </div>
          <button className="primary-button" onClick={run} disabled={running || !activeTopic.trim()}>
            {running ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            {running ? t.running : t.start}
          </button>
        </section>

        {error && <div className="error-line">{error}</div>}

        <section className="results-grid">
          <StrategyPanel
            title={t.single}
            icon={<Search size={18} />}
            segments={singleSegments}
            finalContent={result?.single_agent?.content}
            empty={t.empty}
            hint={t.streamingHint}
            finalLabel={t.final}
          />
          <StrategyPanel
            title={t.adversarial}
            icon={<Activity size={18} />}
            segments={adversarialSegments}
            finalContent={result?.adversarial?.content}
            empty={t.empty}
            hint={t.streamingHint}
            finalLabel={t.final}
          />
        </section>

        <section className="lower-grid">
          <Timeline title={t.timeline} stages={stages} agents={agents} />
          <MetricsPanel title={t.metrics} result={result} onExport={exportJson} exportLabel={t.export} />
          <SourcesPanel title={t.sources} sources={result?.sources || []} />
        </section>
      </section>
    </main>
  );
}

function segmentKey(payload) {
  return `${payload.panel}-${payload.stage}-${payload.role}`;
}

function Segmented({ label, value, options, onChange }) {
  return (
    <div className="segmented-block">
      <span>{label}</span>
      <div className="segmented">
        {options.map((option) => (
          <button key={option.value} className={value === option.value ? "active" : ""} onClick={() => onChange(option.value)}>
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function StrategyPanel({ title, icon, segments, finalContent, empty, hint, finalLabel }) {
  const hasSegments = segments.length > 0;
  const finalAlreadyShown = segments.some((segment) => segment.content === finalContent);
  return (
    <article className="strategy-panel">
      <div className="panel-heading">
        {icon}
        <h2>{title}</h2>
      </div>
      {!hasSegments && <p className="muted-line">{empty}</p>}
      {hasSegments && <p className="streaming-hint">{hint}</p>}
      <div className="stream-panel">
        {segments.map((segment) => (
          <section key={segment.key} className={`stream-segment ${segment.status === "running" ? "running" : ""}`}>
            <div className="segment-meta">
              <span>{formatStage(segment.stage)}</span>
              <strong>{formatRole(segment.role)}</strong>
            </div>
            <p>{segment.content || "..."}</p>
          </section>
        ))}
        {finalContent && !finalAlreadyShown && (
          <section className="stream-segment final">
            <div className="segment-meta">
              <span>{finalLabel}</span>
              <strong>{title}</strong>
            </div>
            <p>{finalContent}</p>
          </section>
        )}
      </div>
    </article>
  );
}

function formatStage(stage) {
  const map = {
    single_agent: "Single",
    round_1: "Round 1",
    round_2: "Round 2",
    synthesis: "Synthesis",
    final: "Final",
  };
  return map[stage] || stage;
}

function formatRole(role) {
  const map = {
    single_agent: "Single Agent",
    pro_logic: "Pro A Logic",
    pro_data: "Pro B Evidence",
    con_logic: "Con A Logic",
    con_data: "Con B Evidence",
    adversarial_synthesis: "Synthesis",
  };
  return map[role] || role;
}

function Timeline({ title, stages, agents }) {
  return (
    <article className="info-panel timeline-panel">
      <div className="panel-heading">
        <Activity size={18} />
        <h2>{title}</h2>
      </div>
      <div className="timeline-list">
        {stages.map((stage, index) => (
          <div key={`${stage.stage}-${index}`} className="timeline-item">
            <CheckCircle2 size={16} />
            <span>{stage.message || stage.stage}</span>
          </div>
        ))}
        {agents.map((agent, index) => (
          <details key={`${agent.role}-${index}`} className="agent-detail">
            <summary>{agent.stage ? `${agent.stage} · ${agent.role}` : agent.role}</summary>
            <p>{agent.content}</p>
          </details>
        ))}
      </div>
    </article>
  );
}

function MetricsPanel({ title, result, onExport, exportLabel }) {
  const metrics = result?.metrics;
  return (
    <article className="info-panel">
      <div className="panel-heading">
        <Activity size={18} />
        <h2>{title}</h2>
      </div>
      <dl className="metric-list">
        <Metric label="Total sec" value={metrics?.total_duration_sec} />
        <Metric label="Single sec" value={metrics?.single_duration_sec} />
        <Metric label="Adv sec" value={metrics?.adversarial_duration_sec} />
        <Metric label="Tokens" value={metrics?.token_estimate} />
        <Metric label="Sources" value={metrics?.source_count} />
        <Metric label="Single diversity" value={metrics?.single_diversity} />
        <Metric label="Adv diversity" value={metrics?.adversarial_diversity} />
      </dl>
      <button className="secondary-button" disabled={!result} onClick={onExport}>
        <Download size={16} />
        {exportLabel}
      </button>
    </article>
  );
}

function Metric({ label, value }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value ?? "·"}</dd>
    </>
  );
}

function SourcesPanel({ title, sources }) {
  return (
    <article className="info-panel source-panel">
      <div className="panel-heading">
        <Search size={18} />
        <h2>{title}</h2>
      </div>
      <div className="source-list">
        {sources.length === 0 && <p className="muted-line">No sources yet.</p>}
        {sources.map((source) => (
          <a key={source.url} href={source.url} target="_blank" rel="noreferrer">
            <strong>{source.title}</strong>
            <span>{source.snippet}</span>
          </a>
        ))}
      </div>
    </article>
  );
}
