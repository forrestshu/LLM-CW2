import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Download,
  Languages,
  Loader2,
  Play,
  ShieldCheck,
} from "lucide-react";
import { fetchHealth, fetchTopics, streamGeneration } from "./api.js";

const COPY = {
  en: {
    title: "Multi-Agent Argument Generator",
    subtitle: "Single-pass logic compared with a local four-round multi-agent workflow.",
    topic: "Motion",
    custom: "Custom motion",
    side: "Target side",
    pro: "Pro",
    con: "Con",
    start: "Generate",
    running: "Generating",
    healthReady: "Local model",
    healthOffline: "Ollama offline",
    single: "Single Agent",
    adversarial: "Multi-Agent",
    final: "Final",
    workflow: "Workflow",
    expandRound: "Expand round",
    collapseRound: "Collapse round",
    metrics: "Metrics",
    candidates: "Candidates",
    eliminated: "Eliminated",
    optimized: "Optimized",
    scorers: "Scoring agents",
    finalScore: "Final avg",
    latencyRatio: "Latency ratio",
    export: "Export JSON",
    useCustom: "Use custom",
    useCache: "Cache",
    empty: "Run once to compare both strategies.",
    streamingHint: "Agent output appears here while the workflow runs.",
    reason: "Reason",
    logicChain: "Logic chain",
  },
  zh: {
    title: "多 Agent 论点生成系统",
    subtitle: "单次逻辑生成与本地四轮多 Agent 流程的并排比较。",
    topic: "辩题",
    custom: "自定义辩题",
    side: "目标立场",
    pro: "正方",
    con: "反方",
    start: "开始生成",
    running: "生成中",
    healthReady: "本地模型",
    healthOffline: "Ollama 未连接",
    single: "单 Agent",
    adversarial: "多 Agent",
    final: "最终结果",
    workflow: "流程",
    expandRound: "展开该轮",
    collapseRound: "折叠该轮",
    metrics: "指标",
    candidates: "候选数",
    eliminated: "淘汰数",
    optimized: "优化数",
    scorers: "评分 Agent",
    finalScore: "最终均分",
    latencyRatio: "耗时倍率",
    export: "导出 JSON",
    useCustom: "使用自定义",
    useCache: "缓存",
    empty: "运行一次后对比两种策略。",
    streamingHint: "工作流运行时，Agent 输出会显示在这里。",
    reason: "理由",
    logicChain: "逻辑链条",
  },
};

export default function App() {
  const [language, setLanguage] = useState("zh");
  const [topics, setTopics] = useState([]);
  const [topicId, setTopicId] = useState("");
  const [customTopic, setCustomTopic] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [targetSide, setTargetSide] = useState("pro");
  const [useCache, setUseCache] = useState(true);
  const [health, setHealth] = useState(null);
  const [running, setRunning] = useState(false);
  const [stages, setStages] = useState([]);
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
  const modelReady = Boolean(health?.ollama?.available);

  async function run() {
    if (!activeTopic.trim()) return;
    setRunning(true);
    setStages([]);
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
          use_cache: useCache,
          side_claim: useCustom
            ? undefined
            : targetSide === "pro"
              ? selectedTopic?.pro_label
              : selectedTopic?.con_label,
        },
        {
          stage: (payload) => setStages((items) => [...items, payload]),
          agent_start: (payload) => startSegment(payload),
          token: (payload) => appendToken(payload),
          agent_done: (payload) => finishSegment(payload),
          panel_append: (payload) => appendCachedSegment(payload),
          final: (payload) => {
            setResult(payload);
            setRunning(false);
          },
          error: (payload) => {
            setError(payload.message || "Generation failed");
            setRunning(false);
          },
          done: () => setRunning(false),
        },
      );
    } catch (err) {
      setError(err.message);
      setRunning(false);
    } finally {
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
            <span className={modelReady ? "status good" : "status warn"}>
              <ShieldCheck size={16} />
              {modelReady ? `${t.healthReady}: ${health?.model || ""}` : t.healthOffline}
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
            <Toggle label={t.useCache} checked={useCache} onChange={setUseCache} />
          </div>
          <button className="primary-button" onClick={run} disabled={running || !activeTopic.trim()}>
            {running ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            {running ? t.running : t.start}
          </button>
        </section>

        {error && <div className="error-line">{error}</div>}

        <section className="results-grid">
          <div className="single-column">
            <StrategyPanel
              variant="single"
              title={t.single}
              icon={<ShieldCheck size={18} />}
              segments={singleSegments}
              finalArgument={result?.single_agent?.argument}
              finalContent={result?.single_agent?.content}
              empty={t.empty}
              hint={t.streamingHint}
              finalLabel={t.final}
              labels={{ reason: t.reason, logicChain: t.logicChain }}
              language={language}
            />
            <MetricsPanel
              title={t.metrics}
              result={result}
              onExport={exportJson}
              exportLabel={t.export}
              labels={{
                candidates: t.candidates,
                eliminated: t.eliminated,
                optimized: t.optimized,
                scorers: t.scorers,
                finalScore: t.finalScore,
                latencyRatio: t.latencyRatio,
              }}
            />
          </div>
          <MultiAgentWorkflowPanel
            title={t.adversarial}
            workflowLabel={t.workflow}
            icon={<Activity size={18} />}
            stages={stages.filter((stage) => stage.stage !== "single_agent" && stage.stage !== "cache")}
            segments={adversarialSegments}
            running={running}
            finalArgument={result?.adversarial?.argument}
            finalContent={result?.adversarial?.content}
            empty={t.empty}
            hint={t.streamingHint}
            finalLabel={t.final}
            labels={{ reason: t.reason, logicChain: t.logicChain }}
            language={language}
            collapseLabels={{ expand: t.expandRound, collapse: t.collapseRound }}
          />
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

const MULTI_AGENT_STAGE_ORDER = [
  "round_1",
  "round_2",
  "elimination",
  "round_3",
  "round_4",
  "final_selection",
  "final",
];

function StrategyPanel({
  variant = "default",
  title,
  icon,
  segments,
  finalArgument,
  finalContent,
  empty,
  hint,
  finalLabel,
  labels,
  language,
}) {
  const hasSegments = segments.length > 0;
  const finalAlreadyShown = segments.some((segment) => segment.content === finalContent);
  return (
    <article className={`strategy-panel ${variant === "single" ? "single-panel" : ""}`}>
      <div className="panel-heading">
        {icon}
        <h2>{title}</h2>
      </div>
      {!hasSegments && !finalArgument && <p className="muted-line">{empty}</p>}
      {hasSegments && <p className="streaming-hint">{hint}</p>}
      <div className={`stream-panel ${variant === "single" ? "stream-panel-compact" : ""}`}>
        {segments.map((segment) => (
          <StreamSegment key={segment.key} segment={segment} language={language} />
        ))}
        {finalContent && !finalAlreadyShown && !finalArgument && (
          <section className="stream-segment final">
            <div className="segment-meta">
              <span>{finalLabel}</span>
              <strong>{title}</strong>
            </div>
            <p>{finalContent}</p>
          </section>
        )}
      </div>
      {finalArgument && (
        <section className="final-argument-summary">
          <div className="segment-meta">
            <span>{finalLabel}</span>
            <strong>{title}</strong>
          </div>
          <ArgumentResult argument={finalArgument} labels={labels} />
        </section>
      )}
    </article>
  );
}

function MultiAgentWorkflowPanel({
  title,
  workflowLabel,
  icon,
  stages,
  segments,
  running,
  finalArgument,
  finalContent,
  empty,
  hint,
  finalLabel,
  labels,
  language,
  collapseLabels,
}) {
  const [collapsedStages, setCollapsedStages] = useState(() => new Set());
  const stageMessages = useMemo(
    () => Object.fromEntries(stages.map((stage) => [stage.stage, stage.message])),
    [stages],
  );
  const groupedSegments = useMemo(() => groupSegmentsByStage(segments), [segments]);
  const activeStages = useMemo(() => {
    const seen = new Set([...stages.map((stage) => stage.stage), ...segments.map((segment) => segment.stage)]);
    return MULTI_AGENT_STAGE_ORDER.filter((stage) => seen.has(stage));
  }, [stages, segments]);
  const hasContent = segments.length > 0 || stages.length > 0 || finalArgument;

  useEffect(() => {
    if (!hasContent && !running) {
      setCollapsedStages(new Set());
    }
  }, [hasContent, running]);

  useEffect(() => {
    if (!activeStages.length) return;
    const latest = activeStages[activeStages.length - 1];
    setCollapsedStages((prev) => {
      if (!prev.has(latest)) return prev;
      const next = new Set(prev);
      next.delete(latest);
      return next;
    });
  }, [activeStages]);

  function toggleStage(stage) {
    setCollapsedStages((prev) => {
      const next = new Set(prev);
      if (next.has(stage)) next.delete(stage);
      else next.add(stage);
      return next;
    });
  }

  return (
    <article className="strategy-panel multi-panel">
      <div className="panel-heading">
        {icon}
        <h2>{title}</h2>
      </div>
      {!hasContent && !running && <p className="muted-line">{empty}</p>}
      {(hasContent || running) && <p className="streaming-hint">{hint}</p>}
      <div className="workflow-timeline">
        {activeStages.map((stage) => {
          const isCollapsed = collapsedStages.has(stage);
          const stageSegments = groupedSegments[stage] || [];
          return (
            <section key={stage} className={`workflow-round ${isCollapsed ? "is-collapsed" : ""}`}>
              <header
                className="round-header"
                role="button"
                tabIndex={0}
                aria-expanded={!isCollapsed}
                aria-label={isCollapsed ? collapseLabels.expand : collapseLabels.collapse}
                onClick={() => toggleStage(stage)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    toggleStage(stage);
                  }
                }}
              >
                {isCollapsed ? (
                  <ChevronRight size={18} className="round-chevron" />
                ) : (
                  <ChevronDown size={18} className="round-chevron" />
                )}
                <CheckCircle2 size={16} />
                <div>
                  <span className="round-label">{workflowLabel}</span>
                  <strong>{stageMessages[stage] || formatStage(stage, language)}</strong>
                </div>
              </header>
              {!isCollapsed && (
                <>
                  <div className="round-scroll-buffer" aria-hidden="true" />
                  <div className="round-body">
                    {stageSegments.map((segment) => (
                      <StreamSegment key={segment.key} segment={segment} language={language} />
                    ))}
                    {running && !stageSegments.length && (
                      <p className="round-pending">
                        <Loader2 className="spin" size={14} />
                        ...
                      </p>
                    )}
                  </div>
                </>
              )}
            </section>
          );
        })}
        {running && !activeStages.length && (
          <section className="workflow-round">
            <header className="round-header round-header-static">
              <Loader2 className="spin" size={16} />
              <strong>{formatStage("round_1", language)}</strong>
            </header>
          </section>
        )}
      </div>
      {finalArgument && (
        <section className="final-argument-summary">
          <div className="segment-meta">
            <span>{finalLabel}</span>
            <strong>{title}</strong>
          </div>
          <ArgumentResult argument={finalArgument} labels={labels} />
        </section>
      )}
      {finalContent && !finalArgument && (
        <section className="final-argument-summary">
          <div className="segment-meta">
            <span>{finalLabel}</span>
            <strong>{title}</strong>
          </div>
          <p className="final-fallback">{finalContent}</p>
        </section>
      )}
    </article>
  );
}

function StreamSegment({ segment, language }) {
  return (
    <section className={`stream-segment ${segment.status === "running" ? "running" : ""}`}>
      <div className="segment-meta">
        <span>{formatStage(segment.stage, language)}</span>
        <strong>{formatRole(segment.role, language)}</strong>
      </div>
      <p>{segment.content || "..."}</p>
    </section>
  );
}

function groupSegmentsByStage(segments) {
  return segments.reduce((groups, segment) => {
    if (!groups[segment.stage]) groups[segment.stage] = [];
    groups[segment.stage].push(segment);
    return groups;
  }, {});
}

function ArgumentResult({ argument, labels }) {
  return (
    <div className="argument-result">
      <div>
        <span>{labels.reason}</span>
        <p>{argument.reason}</p>
      </div>
      <div>
        <span>{labels.logicChain}</span>
        <p>{argument.logic_chain}</p>
      </div>
    </div>
  );
}

function formatStage(stage, language = "en") {
  const maps = {
    en: {
      single_agent: "Single",
      round_1: "Round 1",
      round_2: "Round 2",
      elimination: "Elimination",
      round_3: "Round 3",
      round_4: "Round 4",
      final_selection: "Selection",
      final: "Final",
    },
    zh: {
      single_agent: "单 Agent",
      round_1: "第一轮",
      round_2: "第二轮",
      elimination: "淘汰投票",
      round_3: "第三轮",
      round_4: "第四轮",
      final_selection: "终选",
      final: "最终",
    },
  };
  return maps[language]?.[stage] || maps.en[stage] || stage;
}

function formatRole(role, language = "en") {
  const maps = {
    en: {
      single_agent: "Single Agent",
      argument_generator: "Generator",
      vote_counter: "Vote Counter",
      optimized_pool: "Optimized Pool",
      score_aggregator: "Score Aggregator",
      multi_agent_final: "Multi-Agent Final",
    },
    zh: {
      single_agent: "单 Agent",
      argument_generator: "立论生成",
      vote_counter: "投票统计",
      optimized_pool: "优化池",
      score_aggregator: "分数汇总",
      multi_agent_final: "多 Agent 终稿",
    },
  };
  const labels = maps[language] || maps.en;
  if (role?.startsWith("opposition_")) {
    const index = role.replace("opposition_", "");
    return language === "zh" ? `反方 ${index}` : `Opposition ${index}`;
  }
  if (role?.startsWith("optimizer_")) {
    const id = role.replace("optimizer_", "");
    return language === "zh" ? `优化 ${id}` : `Optimizer ${id}`;
  }
  if (role?.startsWith("scoring_")) {
    const index = role.replace("scoring_", "");
    return language === "zh" ? `评分 ${index}` : `Scoring ${index}`;
  }
  return labels[role] || role;
}

function MetricsPanel({ title, result, onExport, exportLabel, labels }) {
  const metrics = result?.metrics;
  return (
    <article className="info-panel metrics-panel">
      <div className="panel-heading">
        <Activity size={18} />
        <h2>{title}</h2>
      </div>
      <dl className="metric-list">
        <Metric label="Total sec" value={metrics?.total_duration_sec} />
        <Metric label="Single sec" value={metrics?.single_duration_sec} />
        <Metric label="Multi sec" value={metrics?.adversarial_duration_sec} />
        <Metric label="Tokens" value={metrics?.token_estimate} />
        <Metric label={labels.candidates} value={metrics?.candidate_count} />
        <Metric label={labels.eliminated} value={metrics?.eliminated_count} />
        <Metric label={labels.optimized} value={metrics?.optimized_count} />
        <Metric label={labels.scorers} value={metrics?.scoring_agent_count} />
        <Metric label={labels.finalScore} value={metrics?.final_average_score} />
        <Metric label={labels.latencyRatio} value={metrics?.latency_cost_ratio} />
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
