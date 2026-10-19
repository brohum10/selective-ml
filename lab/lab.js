'use strict';
const $ = id => document.getElementById(id);
const pct = value => value === null ? 'n/a' : `${(value*100).toFixed(2)}%`;
const pp = value => `${value>=0?'+':''}${(value*100).toFixed(2)} pp`;
let token, result, selectedSeed = 42, activePanel = 'gate', polling = false;

function node(tag, text, className) {
  const element=document.createElement(tag);
  if(text!==undefined) element.textContent=text;
  if(className) element.className=className;
  return element;
}
function clear(id) { const target=$(id); target.replaceChildren(); return target; }
function stat(label, value, note) {
  const item=node('div',undefined,'stat');
  item.append(node('span',label),node('strong',value));
  if(note) item.append(node('small',note));
  return item;
}
function showPanel(name) {
  activePanel=name;
  for(const button of document.querySelectorAll('[data-panel]')) button.setAttribute('aria-pressed',String(button.dataset.panel===name));
  for(const panel of ['gate','policy','evidence']) $(`${panel}-panel`).hidden=panel!==name;
}
function renderRun(seed) {
  selectedSeed=seed;
  const run=result.runs.find(row=>row.seed===seed), overall=run.gate.slices[0];
  $('selected-label').textContent=`SEED ${seed} / ${run.n} IDENTICAL CLEAN CASES`;
  $('decision-title').textContent=run.gate.decision==='pass'?'Configured checks passed.':run.gate.blockers.some(row=>row.quality==='regression')?'A class-level regression was detected.':'More evidence required.';
  const badge=$('decision-badge'); badge.textContent=run.gate.decision.toUpperCase(); badge.className=`badge ${run.gate.decision}`;
  const stats=node('div',undefined,'stats');
  stats.append(stat('LINEAR V1 ACCURACY',pct(run.baseline_accuracy)),stat('NEURAL V2 ACCURACY',pct(run.candidate_accuracy)));
  clear('gate-summary').append(stats,node('p',`Overall change: ${pp(overall.mean_delta)}. Approx. 95% interval: [${pp(overall.interval_95[0])}, ${pp(overall.interval_95[1])}].`,'fine'));
  const blockers=clear('blockers');
  if(run.gate.blockers.length) {
    for(const blocker of run.gate.blockers) {
      const slice=run.gate.slices.find(row=>row.slice===blocker.slice);
      const box=node('div',undefined,'callout');
      const name=slice.slice.replace('tag:digit-','Digit ');
      box.append(node('p',`${name.toUpperCase()} / ${slice.n} CASES`,'eyebrow'));
      box.append(node('h3',slice.quality_status==='inconclusive'?'Inconclusive—not a proven regression.':slice.quality_status.replaceAll('_',' ')));
      if(slice.interval_95) box.append(node('p',`[${pp(slice.interval_95[0])}, ${pp(slice.interval_95[1])}]`,'interval'));
      box.append(node('p',`Allowed loss: ${pp(-result.protocol.gate.margin)}. This slice does not establish the configured non-regression requirement.`));
      box.append(node('p','The workflow holds the candidate. It does not claim the model is worse.','fine'));
      blockers.append(box);
    }
  } else {
    const box=node('div',undefined,'callout pass-note');
    box.append(node('h3','Overall and every class slice passed.'),node('p','This is evidence under the configured checks, not a guarantee of deployment safety.'));
    blockers.append(box);
  }
  const slices=clear('slices');
  for(const slice of run.gate.slices) { const line=node('div'); line.append(node('span',slice.slice),node('span',`${slice.quality_status} / n=${slice.n}`)); slices.append(line); }
  const policy=run.policy.candidate, model=run.models.candidate;
  const policyStats=node('div',undefined,'stats');
  policyStats.append(stat('DEFERRED FOR REVIEW',`${policy.deferred} / ${policy.total}`,`Learned threshold: ${model.threshold.toFixed(3)}`),stat('ERRORS STILL ACCEPTED',String(policy.accepted_errors),`${policy.deferred_errors} errors deferred; ${policy.deferred_correct} correct answers deferred.`));
  clear('policy-summary').append(policyStats);
  const review=clear('review-example');
  const example=run.examples.find(row=>row.candidate.decision==='review');
  if(example) {
    const card=node('div',undefined,'review-card'), digit=node('div',undefined,'digit');
    digit.setAttribute('role','img'); digit.setAttribute('aria-label',`Public digit row ${example.row}, true label ${example.label}`);
    for(const value of example.pixels) { const pixel=node('i'); pixel.style.backgroundColor=`rgb(${Math.round(14+value/16*88)} ${Math.round(22+value/16*207)} ${Math.round(36+value/16*173)})`; digit.append(pixel); }
    const detail=node('div');
    detail.append(node('p',`FIRST DEFERRED ROW IN THIS SEED / ${example.row}`,'review-label'),node('strong',`Predicted ${example.candidate.prediction}; actual ${example.label}`),node('p',`${pct(example.candidate.confidence)} confidence → REVIEW`),node('p','Unmodified test image. Selection is by row order, not by confidence.','fine'));
    card.append(digit,detail); review.append(card);
  } else {
    review.append(node('p',`This learned policy defers nothing on this split. All ${policy.accepted_errors} errors remain accepted. That limitation is part of the result.`));
  }
  $('provenance').textContent=`Protocol SHA-256\n${result.protocol_sha256}\n\nModel SHA-256 (neural, seed ${seed})\n${run.models.candidate.sha256}\n\nClean-vs-clean control: ${run.control_decision.toUpperCase()}\nComparator: pinned source checksum verified\nCLI comparison: ${run.gate.decision==='pass'?'exit 0':'exit 1'}`;
  showPanel(activePanel);
}
function renderResult(data) {
  result=data;
  const table=clear('runs');
  for(const run of data.runs) {
    const tr=node('tr');
    tr.append(node('td',String(run.seed)),node('td',pct(run.baseline_accuracy)),node('td',pct(run.candidate_accuracy)));
    const decision=node('td'); decision.append(node('span',run.gate.decision.toUpperCase(),`badge ${run.gate.decision}`)); tr.append(decision);
    const action=node('td'), button=node('button',`Inspect ${run.seed}`); button.onclick=()=>{renderRun(run.seed); document.querySelector('.inspector').scrollIntoView({behavior:'auto',block:'start'});}; action.append(button); tr.append(action);table.append(tr);
  }
  const stress=clear('stress-results');
  for(const run of data.runs) stress.append(node('p',`Seed ${run.seed}: ${pct(run.noise_stress.agreement_with_original_labels)} agreement with original labels, σ=${run.noise_stress.sigma}.`));
  $('results').hidden=false; $('download').hidden=false; $('view-results').hidden=false;
  renderRun(selectedSeed);
}
async function poll() {
  if(polling) return;
  polling=true;
  try {
    const response=await fetch('/api/state'); if(!response.ok) throw Error('Could not read experiment state.');
    const state=await response.json();
    $('status').textContent=state.status==='running'?'Running locally':state.status==='complete'?'Complete':state.status==='failed'?'Run failed':'Ready';
    $('run').disabled=state.status==='running';
    $('run').textContent=state.status==='running'?'Training + evaluating…':'Run real experiment ↗';
    $('log-state').textContent=state.status.toUpperCase();
    if(state.events.length) { $('events').textContent=state.events.map(event=>`› ${event.message}`).join('\n'); $('events').scrollTop=$('events').scrollHeight; }
    if(state.status==='complete' && (!result || result.created_utc!==state.result.created_utc)) renderResult(state.result);
    if(state.status==='failed') { $('error').hidden=false; $('error').textContent=state.error; }
    if(state.status==='running') setTimeout(poll,250);
  } catch(error) { $('error').hidden=false; $('error').textContent=error.message; $('run').disabled=false; }
  finally {polling=false;}
}
$('run').onclick=async()=>{
  $('error').hidden=true;
  try {
    if(!token) { const response=await fetch('/api/session'); if(!response.ok) throw Error('Could not initialize local session.'); token=(await response.json()).token; }
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:'{}'});
    if(!response.ok) throw Error((await response.json()).error || 'Could not start experiment.');
    result=undefined; $('results').hidden=true; $('download').hidden=true; $('view-results').hidden=true; await poll();
  } catch(error) { $('error').hidden=false; $('error').textContent=error.message; }
};
for(const button of document.querySelectorAll('[data-panel]')) button.onclick=()=>{showPanel(button.dataset.panel); document.querySelector('.inspector').scrollIntoView({behavior:'auto',block:'start'});};
$('view-results').onclick=()=>$('results').scrollIntoView({behavior:'auto',block:'start'});
poll();
