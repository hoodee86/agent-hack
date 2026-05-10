const API_BASE = '/api';

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => {
    switch (char) {
      case '&':
        return '&amp;';
      case '<':
        return '&lt;';
      case '>':
        return '&gt;';
      case '"':
        return '&quot;';
      case "'":
        return '&#39;';
      default:
        return char;
    }
  });
}

function formatDate(value) {
  if (!value) {
    return '—';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return new Intl.DateTimeFormat('zh-CN', {
    dateStyle: 'medium',
    timeStyle: 'medium',
  }).format(date);
}

function formatRelativeTime(value) {
  if (!value) {
    return '—';
  }

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  const diffMs = date.getTime() - Date.now();
  const rtf = new Intl.RelativeTimeFormat('zh-CN', { numeric: 'auto' });
  const units = [
    ['day', 24 * 60 * 60 * 1000],
    ['hour', 60 * 60 * 1000],
    ['minute', 60 * 1000],
    ['second', 1000],
  ];

  for (const [unit, size] of units) {
    if (Math.abs(diffMs) >= size || unit === 'second') {
      return rtf.format(Math.round(diffMs / size), unit);
    }
  }

  return '刚刚';
}

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = text;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();
  document.execCommand('copy');
  textarea.remove();
}

function renderDetailBlock(label, value) {
  return `
    <div class="detail-block">
      <div class="detail-header">
        <p class="detail-label">${escapeHtml(label)}</p>
        <button type="button" class="copy-button">复制</button>
      </div>
      <pre class="detail-body">${escapeHtml(value)}</pre>
    </div>
  `;
}

function handleCopyButtonClick(event) {
  const button = event.target.closest('.copy-button');
  if (!button) {
    return;
  }

  const detailBlock = button.closest('.detail-block');
  const content = detailBlock?.querySelector('.detail-body, .code-block')?.textContent;
  if (!content) {
    return;
  }

  copyText(content)
    .then(() => {
      button.textContent = '已复制';
      button.classList.add('copied');
      window.setTimeout(() => {
        button.textContent = '复制';
        button.classList.remove('copied');
      }, 1400);
    })
    .catch((error) => {
      console.error('Failed to copy text:', error);
      button.textContent = '复制失败';
      window.setTimeout(() => {
        button.textContent = '复制';
      }, 1400);
    });
}

const state = {
  runs: [],
  selectedRunId: null,
  detail: null,
  selectedEventId: null,
};

const elements = {
  healthChip: document.querySelector('#health-chip'),
  runForm: document.querySelector('#run-form'),
  goalInput: document.querySelector('#goal-input'),
  workspaceInput: document.querySelector('#workspace-input'),
  runSubmit: document.querySelector('#run-submit'),
  refreshButton: document.querySelector('#refresh-button'),
  runList: document.querySelector('#run-list'),
  summaryCard: document.querySelector('#summary-card'),
  timeline: document.querySelector('#timeline'),
  timelineCount: document.querySelector('#timeline-count'),
  approvalCard: document.querySelector('#approval-card'),
  eventInspector: document.querySelector('#event-inspector'),
  selectedEventLabel: document.querySelector('#selected-event-label'),
};

function getEventId(event, index) {
  return event.id || event.event_id || `${event.ts || 'no-ts'}:${event.event || event.type || 'event'}:${index}`;
}

function describeEvent(event) {
  const data = event.data || {};
  const eventType = event.event || event.type || 'event';

  switch (eventType) {
    case 'run_start':
      return data.user_goal || 'Run started';
    case 'tool_proposed':
      return data.tool ? `Proposed ${data.tool}` : 'Tool proposed';
    case 'policy_decision':
      return [data.decision, data.tool].filter(Boolean).join(' · ') || 'Policy decision';
    case 'tool_result':
      return [data.tool, data.ok === true ? 'success' : data.ok === false ? 'failed' : null].filter(Boolean).join(' · ') || 'Tool result';
    case 'approval_presented':
    case 'approval_requested':
      return data.impact_summary || data.reason || 'Approval requested';
    case 'run_end':
      return data.status || 'Run completed';
    default: {
      if (typeof data.impact_summary === 'string' && data.impact_summary) {
        return data.impact_summary;
      }

      if (typeof data.reason === 'string' && data.reason) {
        return data.reason;
      }

      if (typeof data.tool === 'string' && data.tool) {
        return data.tool;
      }

      const keys = Object.keys(data);
      return keys.length > 0 ? keys.slice(0, 3).join(' · ') : 'No summary';
    }
  }
}

function normalizeEvent(event, index) {
  const data = event.data || {};
  return {
    id: getEventId(event, index),
    type: event.event || event.type || 'event',
    timestamp: event.ts || event.timestamp || null,
    source: data.tool || data.decision || event.run_id || 'event',
    description: describeEvent(event),
    details: data,
  };
}

function getEvents() {
  return (state.detail?.events || []).map(normalizeEvent);
}

function getFinalAnswer() {
  const summary = state.detail?.summary || {};
  if (summary.final_answer) {
    return summary.final_answer;
  }

  const runEndEvent = (state.detail?.events || []).findLast(
    (event) => (event.event || event.type) === 'run_end'
  );
  const eventFinalAnswer = runEndEvent?.data?.final_answer;
  if (eventFinalAnswer) {
    return eventFinalAnswer;
  }

  return summary.final_answer_preview || '';
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    elements.healthChip.textContent = `Online: ${data.status}`;
    elements.healthChip.className = 'health-chip ok';
  } catch (err) {
    elements.healthChip.textContent = 'Offline';
    elements.healthChip.className = 'health-chip failed';
  }
}

async function fetchRuns() {
  try {
    const res = await fetch(`${API_BASE}/runs`);
    const data = await res.json();
    state.runs = data.runs;
    renderRunList();

    if (state.runs.length > 0 && !state.selectedRunId) {
      selectRun(state.runs[0].run_id);
    } else if (state.runs.length > 0 && state.selectedRunId) {
      await fetchRunDetail(state.selectedRunId);
    }
  } catch (err) {
    console.error('Failed to fetch runs:', err);
  }
}

async function fetchRunDetail(runId) {
  try {
    const res = await fetch(`${API_BASE}/runs/${runId}`);
    state.detail = await res.json();

    const events = getEvents();
    if (events.length === 0) {
      state.selectedEventId = null;
    } else if (!state.selectedEventId || !events.some((event) => event.id === state.selectedEventId)) {
      state.selectedEventId = events[events.length - 1].id;
    }

    renderSummary();
    renderTimeline();
    renderApproval();
    renderInspector();
  } catch (err) {
    console.error(`Failed to fetch detail for ${runId}:`, err);
  }
}

function selectRun(runId) {
  state.selectedRunId = runId;
  state.selectedEventId = null;
  renderRunList();
  fetchRunDetail(runId);
}

function selectEvent(eventId) {
  state.selectedEventId = eventId;
  renderTimeline();
  renderInspector();
}

function renderRunList() {
  elements.runList.innerHTML = '';
  if (state.runs.length === 0) {
    elements.runList.innerHTML = '<div class="empty-state">尚无执行记录。</div>';
    return;
  }

  state.runs.forEach(run => {
    const div = document.createElement('div');
    div.className = `run-item ${run.run_id === state.selectedRunId ? 'active' : ''}`;
    div.onclick = () => selectRun(run.run_id);

    div.innerHTML = `
      <div class="run-title">
        <strong>${escapeHtml(run.user_goal || run.run_id)}</strong>
      </div>
      <div class="run-meta">
        <span class="status-pill ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span>
        <span title="${escapeHtml(formatDate(run.updated_at))}">${escapeHtml(formatRelativeTime(run.updated_at))}</span>
      </div>
    `;
    elements.runList.appendChild(div);
  });
}

function renderSummary() {
  const summary = state.detail?.summary;
  if (!summary) {
    elements.summaryCard.innerHTML = '<div class="empty-state">选择一个 run 以查看状态。</div>';
    return;
  }

  const finalAnswer = getFinalAnswer();

  elements.summaryCard.innerHTML = `
    <div class="panel-heading">
      <h2>Run Summary</h2>
      <span class="status-pill ${escapeHtml(summary.status)}">${escapeHtml(summary.status)}</span>
    </div>
    <div class="summary-grid">
      <div>
        <dt>Goal</dt>
        <dd>${escapeHtml(summary.user_goal || summary.run_id)}</dd>
      </div>
      <div>
        <dt>Workspace</dt>
        <dd>${escapeHtml(summary.workspace_root || '—')}</dd>
      </div>
      <div>
        <dt>Run ID</dt>
        <dd>${escapeHtml(summary.run_id)}</dd>
      </div>
      <div>
        <dt>Mode</dt>
        <dd>${escapeHtml(summary.mode || 'new')}</dd>
      </div>
      <div>
        <dt>Updated</dt>
        <dd>${escapeHtml(formatDate(summary.updated_at))}</dd>
      </div>
      <div>
        <dt>Iterations / Commands</dt>
        <dd>${escapeHtml(summary.iteration_count ?? '—')} / ${escapeHtml(summary.command_count ?? '—')}</dd>
      </div>
      <div>
        <dt>Last Event</dt>
        <dd>${escapeHtml(summary.last_event || '—')}</dd>
      </div>
      <div>
        <dt>Approval Pending</dt>
        <dd>${summary.approval_pending ? 'Yes' : 'No'}</dd>
      </div>
    </div>
    ${finalAnswer ? renderDetailBlock('Final Answer', finalAnswer) : ''}
    ${summary.error ? renderDetailBlock('Background Error', summary.error) : ''}
  `;
}

function renderTimeline() {
  const events = getEvents();
  elements.timelineCount.textContent = events.length;
  elements.timeline.innerHTML = '';

  if (events.length === 0) {
    elements.timeline.innerHTML = '<div class="empty-state">暂无事件记录</div>';
    return;
  }

  events.forEach(event => {
    const div = document.createElement('div');
    div.className = `event-card ${event.id === state.selectedEventId ? 'active' : ''}`;
    div.onclick = () => selectEvent(event.id);

    div.innerHTML = `
      <div class="event-title">
        <span>${escapeHtml(event.type)}</span>
        <span class="event-pill ${escapeHtml(event.type)}">${escapeHtml(event.source)}</span>
      </div>
      <div class="event-subtitle">${escapeHtml(event.description || '')}</div>
      <div class="event-meta">
        <span>${escapeHtml(formatDate(event.timestamp))}</span>
      </div>
    `;
    elements.timeline.appendChild(div);
  });
}

function renderInspector() {
  const events = getEvents();
  const event = events.find((entry) => entry.id === state.selectedEventId);

  if (!event) {
    elements.selectedEventLabel.textContent = 'None';
    elements.eventInspector.innerHTML = '<div class="empty-state">在左侧时间线中点击事件查看详情</div>';
    return;
  }

  elements.selectedEventLabel.textContent = event.type;

  let detailsHtml = `
    ${renderDetailBlock('Event', event.type)}
    ${renderDetailBlock('Timestamp', formatDate(event.timestamp))}
  `;
  
  if (event.details) {
    Object.entries(event.details).forEach(([key, value]) => {
      let displayValue = value;
      if (typeof value === 'object') {
        displayValue = JSON.stringify(value, null, 2);
      }
      detailsHtml += renderDetailBlock(key, displayValue);
    });
  } else {
      detailsHtml = '<div class="empty-state">无详细负载内容</div>';
  }

  elements.eventInspector.innerHTML = detailsHtml;
}

function renderApproval() {
  const approval = state.detail?.pending_approval;
  if (!approval) {
    elements.approvalCard.style.display = 'none';
    elements.approvalCard.innerHTML = '';
    return;
  }

  elements.approvalCard.style.display = 'block';
  elements.approvalCard.className = 'approval-container';

  elements.approvalCard.innerHTML = `
    <div class="approval-header">
      <h2>⚠️ Needs Approval: ${escapeHtml(approval.tool_summary)}</h2>
      <span class="status-pill paused">${escapeHtml(approval.risk_level || 'review')}</span>
    </div>
    <div class="approval-form">
      <form id="approval-form-action">
        <label>
          <span style="font-size:12px;color:var(--text-muted);">Feedback (Optional)</span>
          <textarea id="approval-feedback" rows="3" placeholder="填写修改意见 / 允许继续"></textarea>
        </label>
        <div class="approval-actions">
          <button type="submit" class="btn-primary" data-action="approve">Approve</button>
          <button type="button" class="reject" data-action="reject">Reject</button>
        </div>
      </form>
    </div>
  `;

  document.getElementById('approval-form-action').addEventListener('submit', handleApproval);
  document.querySelector('.reject').addEventListener('click', () => handleApproval(new Event('submit'), 'reject'));
}

async function handleApproval(e, explicitAction) {
  e.preventDefault();
  if (!state.selectedRunId) return;

  const action = explicitAction || e.target.querySelector('button[type="submit"]').dataset.action;
  const feedback = document.getElementById('approval-feedback').value;

  try {
    const res = await fetch(`${API_BASE}/runs/${state.selectedRunId}/approval`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: action, note: feedback || undefined })
    });
    
    if (res.ok) {
        alert(action === 'approve' ? '审批通过' : '已拒绝');
        fetchRunDetail(state.selectedRunId);
    } else {
        const err = await res.json();
        alert(`提交失败: ${err.detail}`);
    }
  } catch (err) {
    console.error(err);
    alert('提交网络异常');
  }
}

elements.runForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const goal = elements.goalInput.value.trim();
  const workspace = elements.workspaceInput.value.trim() || undefined;

  if (!goal) return;
  elements.runSubmit.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/runs`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ goal, workspace })
    });
    const data = await res.json();
    elements.goalInput.value = '';
    await fetchRuns();
    selectRun(data.run_id);
  } catch (err) {
    console.error(err);
    alert('Failed to start run');
  } finally {
    elements.runSubmit.disabled = false;
  }
});

elements.refreshButton.addEventListener('click', fetchRuns);
document.addEventListener('click', handleCopyButtonClick);

setInterval(checkHealth, 10000);
if (state.selectedRunId) {
    setInterval(() => fetchRunDetail(state.selectedRunId), 5000);
} else {
    setInterval(fetchRuns, 5000);
}

checkHealth();
fetchRuns();
