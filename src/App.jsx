import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import "./App.css";

const API = "https://web-production-e6005.up.railway.app";

// Request Detail Modal
function RequestDetailModal({ request, onClose }) {
  if (!request) return null;

  const statusColor = s => ({
    approved: "#22c55e",
    pending:  "#eab308",
    denied:   "#ef4444",
  }[s] || "#6b7280");

  const statusLabel = s => ({
    approved: "Approved",
    pending:  "Pending",
    denied:   "Denied",
  }[s] || s);

  const typeLabel = t => ({
    pto:     "PTO Request",
    expense: "Expense Report",
  }[t] || t);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box request-detail-box" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        <div className="request-detail-header">
          <div className="request-detail-type">{typeLabel(request.request_type)}</div>
          <div
            className="request-detail-status"
            style={{ color: statusColor(request.status), borderColor: statusColor(request.status) }}
          >
            {statusLabel(request.status)}
          </div>
        </div>
        <div className="request-detail-body">
          <p className="request-detail-description">{request.description}</p>
          <div className="request-detail-meta">
            <div className="request-meta-row">
              <span className="request-meta-label">Submitted</span>
              <span className="request-meta-value">{request.submitted_at}</span>
            </div>
            <div className="request-meta-row">
              <span className="request-meta-label">Type</span>
              <span className="request-meta-value">{typeLabel(request.request_type)}</span>
            </div>
            <div className="request-meta-row">
              <span className="request-meta-label">Status</span>
              <span className="request-meta-value" style={{ color: statusColor(request.status) }}>
                {statusLabel(request.status)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// About Modal
function AboutModal({ onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box about-box" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        <div className="about-header">
          <span className="about-logo">⬡</span>
          <h2>Velo Employee Portal</h2>
          <p className="about-subtitle">Powered by Aria</p>
        </div>
        <div className="about-body">

          <div className="about-section">
            <div className="about-section-title">Project Overview</div>
            <p>
              Aria is an AI-powered enterprise HR agent built as a portfolio project
              to demonstrate how modern AI agent architecture can solve real enterprise
              problems at scale. The application integrates a retrieval augmented
              generation pipeline, a reasoning agent with tool use, vector search,
              and a relational database into a single deployable full stack system.
            </p>
          </div>

          <div className="about-section">
            <div className="about-section-title">Case Example: Velo</div>
            <p>
              Velo is a fictional B2B SaaS company building marketing analytics
              software. With 74 employees across 8 departments and hiring
              aggressively, Velo is scaling faster than its internal knowledge
              systems can keep up with.
            </p>
            <br />
            <p>
              New employees are joining Engineering, Sales, and Customer Success
              every month. Each cohort arrives with the same questions. What is
              my PTO allowance? How do I submit an expense report? What tools do
              I need to set up? What should I be doing in my first 30 days?
            </p>
            <br />
            <p>
              The knowledge to answer all of these questions exists. It lives
              across 13 internal documents including the employee handbook,
              role specific onboarding guides, benefits documentation, and team
              process runbooks. But employees do not know where to look, and
              when they cannot find the answer they ask their manager. Managers
              across all 8 departments are fielding the same repetitive questions
              from every new hire cohort, pulling them away from the work that
              actually drives the business. HR is handling a constant stream of
              basic requests that could be self served. Onboarding is slower and
              more expensive than it needs to be.
            </p>
          </div>

          <div className="about-section">
            <div className="about-section-title">The Solution</div>
            <p>
              Aria gives every Velo employee instant, personalized answers
              without involving a manager or waiting for an HR response.
            </p>
            <br />
            <p>
              Employees ask questions in plain language. Aria determines whether
              the answer lives in a company document or in the employee database,
              queries the right source or both simultaneously, and returns a
              single accurate answer grounded in real data. A new engineer asking
              about their first 30 days gets the engineering onboarding plan. A
              six year VP asking about PTO gets told they are on the unlimited
              tier based on their actual start date pulled directly from the
              database. Every answer is specific to who is asking.
            </p>
            <br />
            <p>
              The agent also surfaces each employee's active PTO requests and
              expense reports directly in the interface, giving employees a
              single place to both ask questions and track their own HR activity.
            </p>
          </div>

          <div className="about-section">
            <div className="about-section-title">How It Works</div>
            <div className="about-features">
              <div className="about-feature">
                <div>
                  <strong>RAG Pipeline</strong>
                  <p>
                    13 internal documents are chunked into 800 character
                    overlapping segments and embedded using OpenAI
                    text-embedding-3-small. Embeddings are stored in ChromaDB
                    with metadata tagging each chunk by document category and
                    relevant employee role. At query time the user's question
                    is embedded and the top 5 most semantically similar chunks
                    are retrieved as grounding context for the LLM.
                  </p>
                </div>
              </div>
              <div className="about-feature">
                <div>
                  <strong>LangGraph ReAct Agent</strong>
                  <p>
                    The agent is built on LangGraph's ReAct architecture. On
                    each turn the LLM reasons about what information it needs,
                    selects a tool, executes it, observes the result, and
                    decides whether to call another tool or return a final
                    answer. This loop allows the agent to combine results from
                    multiple sources in a single response rather than following
                    a fixed retrieval path.
                  </p>
                </div>
              </div>
              <div className="about-feature">
                <div>
                  <strong>Dual Data Sources</strong>
                  <p>
                    Unstructured knowledge lives in ChromaDB as vector
                    embeddings. Structured employee data lives in SQLite via
                    SQLAlchemy and includes employees, departments, and HR
                    requests. The agent treats each as a separate tool and can
                    query both within the same conversation turn, combining
                    policy information from documents with live records from
                    the database.
                  </p>
                </div>
              </div>
              <div className="about-feature">
                <div>
                  <strong>Personalized System Prompt</strong>
                  <p>
                    Every agent session is initialized with a dynamically built
                    system prompt containing the current user's name, role,
                    department, persona type, and today's date. The prompt
                    includes explicit rules for tenure based PTO calculation,
                    role specific document retrieval, and database query
                    patterns. This ensures answers are grounded in who is
                    asking rather than returning generic policy text.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="about-stack">
            <span>LangGraph</span>
            <span>RAG Pipeline</span>
            <span>ChromaDB</span>
            <span>OpenAI GPT-4o</span>
            <span>FastAPI</span>
            <span>SQLite</span>
            <span>SQLAlchemy</span>
            <span>React</span>
            <span>Vite</span>
            <span>Railway</span>
            <span>Netlify</span>
          </div>

        </div>
      </div>
    </div>
  );
}

// Persona Switcher Modal
function PersonaModal({ personas, current, onSelect, onClose }) {
  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-box persona-box" onClick={e => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>✕</button>
        <h3 className="persona-modal-title">Switch Employee</h3>
        <p className="persona-modal-sub">Select a profile to experience the portal as a different Velo employee</p>
        <div className="persona-grid">
          {personas.map(p => (
            <button
              key={p.id}
              className={`persona-card ${current?.id === p.id ? "active" : ""}`}
              onClick={() => { onSelect(p); onClose(); }}
            >
              <div className="persona-card-name">{p.name}</div>
              <div className="persona-card-role">{p.role}</div>
              <div className="persona-card-dept">{p.department}</div>
              <div className="persona-card-tagline">{p.tagline}</div>
              {current?.id === p.id && <div className="persona-active-badge">Active</div>}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// Sidebar
function Sidebar({ profile, ptoRequests, expenseRequests, onSwitchClick, onRequestClick }) {
  if (!profile) return (
    <aside className="sidebar">
      <div className="sidebar-loading">Loading profile...</div>
    </aside>
  );

  const statusColor = s => ({
    approved: "#22c55e",
    pending:  "#eab308",
    denied:   "#ef4444",
  }[s] || "#6b7280");

  const statusLabel = s => ({
    approved: "Approved",
    pending:  "Pending",
    denied:   "Denied",
  }[s] || s);

  const RequestItem = ({ r }) => (
    <div className="ticket-item clickable" onClick={() => onRequestClick(r)}>
      <div className="ticket-dot" style={{ background: statusColor(r.status) }} />
      <div className="ticket-info">
        <div className="ticket-title">{r.description.slice(0, 48)}{r.description.length > 48 ? "..." : ""}</div>
        <div className="ticket-meta">{statusLabel(r.status)} • {r.submitted_at}</div>
      </div>
      <div className="ticket-chevron">›</div>
    </div>
  );

  return (
    <aside className="sidebar" style={{ width: "100%" }}>
      <div className="sidebar-profile">
        <div className="profile-avatar">
          {profile.name.split(" ").map(n => n[0]).join("")}
        </div>
        <div className="profile-info">
          <div className="profile-name">{profile.name}</div>
          <div className="profile-role">{profile.role}</div>
          <div className="profile-dept">{profile.department}</div>
        </div>
        <div className="profile-meta">
          <div className="meta-item">
            <span className="meta-label">Email</span>
            <span className="meta-value">{profile.email}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Started</span>
            <span className="meta-value">{profile.start_date}</span>
          </div>
          <div className="meta-item">
            <span className="meta-label">Level</span>
            <span className="meta-value">{profile.is_manager ? "Manager" : "Individual Contributor"}</span>
          </div>
        </div>
      </div>

      {ptoRequests.length > 0 && (
        <div className="sidebar-section">
          <div className="section-header">
            <span className="section-title">PTO Requests</span>
            <span className="section-badge">{ptoRequests.length}</span>
          </div>
          <div className="ticket-list">
            {ptoRequests.map(r => <RequestItem key={r.id} r={r} />)}
          </div>
        </div>
      )}

      {expenseRequests.length > 0 && (
        <div className="sidebar-section">
          <div className="section-header">
            <span className="section-title">Expense Reports</span>
            <span className="section-badge">{expenseRequests.length}</span>
          </div>
          <div className="ticket-list">
            {expenseRequests.map(r => <RequestItem key={r.id} r={r} />)}
          </div>
        </div>
      )}

      <button className="switch-btn" onClick={onSwitchClick}>
        Switch Employee
      </button>
    </aside>
  );
}

// Chat
function Chat({ persona, profile }) {
  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    if (!profile) return;
    const firstName = profile.name.split(" ")[0];
    setMessages([{
      role:    "assistant",
      content: `Hi ${firstName}! I'm Aria, your Velo knowledge assistant. I can help you with company policies, onboarding steps, benefits, PTO, expense reporting, team processes, and anything else you need to know about working at Velo.\n\nThis portal includes 4 employee profiles you can switch between using the Switch Employee button on the left — a new engineering hire, a sales manager, an HR operations lead, and a VP of Customer Success. Each persona has different tenure, role-specific documents, and their own PTO and expense requests. Try asking the same question as different employees to see how answers change.\n\nWhat can I help you with today?`,
    }]);
    setInput("");
  }, [profile?.name]);

  const send = async () => {
    const q = input.trim();
    if (!q || loading) return;

    setMessages(prev => [...prev, { role: "user", content: q }]);
    setInput("");
    setLoading(true);

    try {
      const history = messages.slice(1).map(m => ({
        role:    m.role,
        content: m.content,
      }));

      const res = await axios.post(`${API}/api/chat`, {
        question:     q,
        persona:      persona.id,
        chat_history: history,
      });
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: res.data.answer,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role:    "assistant",
        content: "I'm having trouble connecting to the server. Please make sure the backend is running.",
        error:   true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKey = e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  useEffect(() => {
    const el = document.querySelector(".messages");
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, loading]);

  const formatMessage = text => {
    return text
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br/>");
  };

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`message ${m.role} ${m.error ? "error" : ""}`}>
            {m.role === "assistant" && (
              <div className="message-avatar">A</div>
            )}
            <div
              className="message-bubble"
              dangerouslySetInnerHTML={{ __html: formatMessage(m.content) }}
            />
            {m.role === "user" && (
              <div className="message-avatar user-avatar">
                {profile?.name.split(" ").map(n => n[0]).join("") || "U"}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-avatar">A</div>
            <div className="message-bubble typing">
              <span /><span /><span />
            </div>
          </div>
        )}
      </div>

      <div className="input-area">
        <textarea
          className="input-box"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKey}
          placeholder="Ask Aria anything about Velo..."
          rows={1}
          disabled={loading}
        />
        <button
          className={`send-btn ${loading ? "loading" : ""}`}
          onClick={send}
          disabled={loading || !input.trim()}
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}

// App
export default function App() {
  const [personas,        setPersonas]        = useState([]);
  const [currentPersona,  setCurrentPersona]  = useState(null);
  const [profile,         setProfile]         = useState(null);
  const [ptoRequests,     setPtoRequests]     = useState([]);
  const [expenseRequests, setExpenseRequests] = useState([]);
  const [showAbout,       setShowAbout]       = useState(false);
  const [showPersonas,    setShowPersonas]    = useState(false);
  const [selectedRequest, setSelectedRequest] = useState(null);

  // Resizable sidebar
  const [sidebarWidth, setSidebarWidth] = useState(280);
  const isResizing  = useRef(false);
  const startX      = useRef(0);
  const startWidth  = useRef(0);

  const onMouseDown = useCallback(e => {
    isResizing.current  = true;
    startX.current      = e.clientX;
    startWidth.current  = sidebarWidth;
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";
  }, [sidebarWidth]);

  useEffect(() => {
    const onMouseMove = e => {
      if (!isResizing.current) return;
      const delta    = e.clientX - startX.current;
      const newWidth = Math.min(480, Math.max(200, startWidth.current + delta));
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      isResizing.current             = false;
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
    };
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup",   onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup",   onMouseUp);
    };
  }, []);

  useEffect(() => {
    axios.get(`${API}/api/personas`)
      .then(r => {
        setPersonas(r.data.personas);
        setCurrentPersona(r.data.personas[0]);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!currentPersona) return;
    setProfile(null);
    setPtoRequests([]);
    setExpenseRequests([]);

    axios.get(`${API}/api/user/${currentPersona.id}`)
      .then(r => setProfile(r.data))
      .catch(console.error);

    axios.get(`${API}/api/hr_requests/${currentPersona.id}`)
      .then(r => {
        setPtoRequests(r.data.pto_requests     || []);
        setExpenseRequests(r.data.expense_requests || []);
      })
      .catch(console.error);
  }, [currentPersona]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-left">
          <span className="topbar-logo">⬡</span>
          <span className="topbar-title">Velo Employee Portal</span>
          <span className="topbar-powered">powered by Aria</span>
        </div>
        <button className="about-btn" onClick={() => setShowAbout(true)}>
          About
        </button>
      </header>

      <main className="main">
        <div style={{ width: sidebarWidth, minWidth: sidebarWidth, display: "flex", flexShrink: 0 }}>
          <Sidebar
            profile={profile}
            ptoRequests={ptoRequests}
            expenseRequests={expenseRequests}
            onSwitchClick={() => setShowPersonas(true)}
            onRequestClick={r => setSelectedRequest(r)}
          />
        </div>

        <div className="resize-handle" onMouseDown={onMouseDown} />

        {currentPersona && (
          <Chat persona={currentPersona} profile={profile} />
        )}
      </main>

      {showAbout    && <AboutModal onClose={() => setShowAbout(false)} />}
      {showPersonas && (
        <PersonaModal
          personas={personas}
          current={currentPersona}
          onSelect={p => setCurrentPersona(p)}
          onClose={() => setShowPersonas(false)}
        />
      )}
      {selectedRequest && (
        <RequestDetailModal
          request={selectedRequest}
          onClose={() => setSelectedRequest(null)}
        />
      )}
    </div>
  );
}
