import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API = "http://localhost:8000";

// ─── About Modal ──────────────────────────────────────────────────────────────
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
          <p>
            Aria is Velo's intelligent HR and internal knowledge assistant. 
            Employees can instantly ask questions about company policies, 
            benefits, PTO, expense reporting, onboarding, and processes — 
            and see their own HR requests in real time.
          </p>
          <div className="about-features">
            <div className="about-feature">
              <span className="feature-icon">📄</span>
              <div>
                <strong>13 Internal HR Documents</strong>
                <p>Searches the employee handbook, PTO policy, expense policy, 
                onboarding guides by role, benefits guide, IT security, 
                team runbooks, and company OKRs.</p>
              </div>
            </div>
            <div className="about-feature">
              <span className="feature-icon">🗄️</span>
              <div>
                <strong>Live Employee Data</strong>
                <p>Queries real-time employee records, department structure, 
                PTO requests, and expense reports directly from the 
                internal database.</p>
              </div>
            </div>
            <div className="about-feature">
              <span className="feature-icon">🤖</span>
              <div>
                <strong>AI Agent Reasoning</strong>
                <p>Built on LangGraph and OpenAI GPT-4o. The agent decides 
                which sources to query — documents, database, or both — 
                and synthesizes a single personalized answer.</p>
              </div>
            </div>
            <div className="about-feature">
              <span className="feature-icon">👤</span>
              <div>
                <strong>Tenure-Aware Personalization</strong>
                <p>Answers are tailored to each employee's role, department, 
                and exact tenure. PTO allowances, onboarding plans, and 
                expense thresholds all reflect the individual — not 
                generic policy.</p>
              </div>
            </div>
          </div>
          <div className="about-stack">
            <span>RAG Pipeline</span>
            <span>LangGraph Agent</span>
            <span>ChromaDB</span>
            <span>FastAPI</span>
            <span>GPT-4o</span>
            <span>SQLite</span>
            <span>React</span>
            <span>Vite</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Persona Switcher Modal ───────────────────────────────────────────────────
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

// ─── Sidebar ──────────────────────────────────────────────────────────────────
function Sidebar({ profile, ptoRequests, expenseRequests, onSwitchClick }) {
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
    <div className="ticket-item">
      <div className="ticket-dot" style={{ background: statusColor(r.status) }} />
      <div className="ticket-info">
        <div className="ticket-title">{r.description.slice(0, 48)}{r.description.length > 48 ? "..." : ""}</div>
        <div className="ticket-meta">{statusLabel(r.status)} • {r.submitted_at}</div>
      </div>
    </div>
  );

  return (
    <aside className="sidebar">
      {/* Profile */}
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

      {/* PTO Requests */}
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

      {/* Expense Requests */}
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

// ─── Chat ─────────────────────────────────────────────────────────────────────
function Chat({ persona, profile }) {
  const [messages, setMessages] = useState([]);
  const [input,    setInput]    = useState("");
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    if (!profile) return;
    const firstName = profile.name.split(" ")[0];
    setMessages([{
      role:    "assistant",
      content: `Hi ${firstName}! I'm Aria, your Velo knowledge assistant. I can help you with company policies, onboarding steps, benefits, PTO, expense reporting, team processes, and anything else you need to know about working at Velo. What can I help you with today?`,
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

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [personas,       setPersonas]       = useState([]);
  const [currentPersona, setCurrentPersona] = useState(null);
  const [profile,        setProfile]        = useState(null);
  const [ptoRequests,    setPtoRequests]    = useState([]);
  const [expenseRequests,setExpenseRequests]= useState([]);
  const [showAbout,      setShowAbout]      = useState(false);
  const [showPersonas,   setShowPersonas]   = useState(false);

  // Load personas on mount
  useEffect(() => {
    axios.get(`${API}/api/personas`)
      .then(r => {
        setPersonas(r.data.personas);
        setCurrentPersona(r.data.personas[0]);
      })
      .catch(console.error);
  }, []);

  // Load profile and HR requests when persona changes
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
        setPtoRequests(r.data.pto_requests || []);
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
        <Sidebar
          profile={profile}
          ptoRequests={ptoRequests}
          expenseRequests={expenseRequests}
          onSwitchClick={() => setShowPersonas(true)}
        />
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
    </div>
  );
}
