'use strict';
const $ = id => document.getElementById(id);
const pct = value => value === null ? 'n/a' : `${(value*100).toFixed(2)}%`;
const pp = value => `${value>=0?'+':''}${(value*100).toFixed(2)} pp`;
let token, result, analysis, selectedSeed = 42, activePanel = 'gate', polling = false, atlasLimit = 12;

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
  for(const panel of ['gate','policy','atlas','sandbox','evidence']) $(`${panel}-panel`).hidden=panel!==name;
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
  renderDiagnostics();
  showPanel(activePanel);
}
function renderResult(data, diagnostics) {
  result=data;
  analysis=diagnostics;
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
    if(state.status==='complete' && (!result || result.created_utc!==state.result.created_utc)) renderResult(state.result,state.diagnostics);
    if(state.status==='failed') { $('error').hidden=false; $('error').textContent=state.error; }
    if(state.status==='running') setTimeout(poll,250);
  } catch(error) { $('error').hidden=false; $('error').textContent=error.message; $('run').disabled=false; }
  finally {polling=false;}
}
$('run').onclick=async()=>{
  $('error').hidden=true;
  $('run').disabled=true;
  try {
    if(!token) { const response=await fetch('/api/session'); if(!response.ok) throw Error('Could not initialize local session.'); token=(await response.json()).token; }
    const response=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':token},body:'{}'});
    if(!response.ok) throw Error((await response.json()).error || 'Could not start experiment.');
    result=undefined; $('results').hidden=true; $('download').hidden=true; $('view-results').hidden=true; await poll();
  } catch(error) { $('error').hidden=false; $('error').textContent=error.message; $('run').disabled=false; }
};
for(const button of document.querySelectorAll('[data-panel]')) button.onclick=()=>{showPanel(button.dataset.panel); document.querySelector('.inspector').scrollIntoView({behavior:'auto',block:'start'});};
$('view-results').onclick=()=>$('results').scrollIntoView({behavior:'auto',block:'start'});

function selectedAnalysis() { return analysis.runs.find(row=>row.seed===selectedSeed); }
function selectedRun() { return result.runs.find(row=>row.seed===selectedSeed); }
function digitImage(example) {
  const digit=node('div',undefined,'digit');
  digit.setAttribute('role','img'); digit.setAttribute('aria-label',`Digit row ${example.row}; true label ${example.label}`);
  for(const value of example.pixels) {
    const pixel=node('i');
    pixel.style.backgroundColor=`rgb(${Math.round(14+value/16*88)} ${Math.round(22+value/16*207)} ${Math.round(36+value/16*173)})`;
    digit.append(pixel);
  }
  return digit;
}
function renderDiagnostics() {
  atlasLimit=12;
  const current=selectedAnalysis(), changes=clear('change-counts');
  for(const [key,label] of [['both_correct','Both correct'],['improved','Improved rows'],['regressed','Regressed rows'],['both_wrong','Both wrong']])
    changes.append(stat(label,String(current.changes[key])));
  renderAtlas();
  const matrix=clear('confusion'), head=node('tr');
  head.append(node('th','True ↓ / Pred →'));
  for(let i=0;i<10;i++) head.append(node('th',String(i)));
  const thead=node('thead'); thead.append(head); matrix.append(thead);
  const tbody=node('tbody');
  for(let i=0;i<10;i++) {
    const row=node('tr'); row.append(node('th',String(i)));
    for(let j=0;j<10;j++) { const cell=node('td',String(current.confusion.candidate[i][j]),i===j?'matrix-correct':current.confusion.candidate[i][j]?'matrix-error':''); cell.setAttribute('aria-label',`True ${i}, predicted ${j}: ${current.confusion.candidate[i][j]} cases`); row.append(cell); }
    tbody.append(row);
  }
  matrix.append(tbody);
  const reliability=clear('reliability');
  reliability.append(node('p',`Baseline ECE: ${pct(current.reliability.baseline.ece)} · Candidate ECE: ${pct(current.reliability.candidate.ece)}`,'fine'));
  const table=node('table',undefined,'reliability-table'), header=node('tr');
  table.append(node('caption','Candidate model · fixed confidence bins'));
  for(const label of ['Confidence bin','Cases','Mean confidence','Accuracy']) header.append(node('th',label));
  const headers=node('thead'); headers.append(header); table.append(headers);
  const body=node('tbody');
  for(const bin of current.reliability.candidate.bins) {
    const row=node('tr');
    for(const text of [`${Math.round(bin.lower*100)}–${Math.round(bin.upper*100)}%${bin.upper===1?' inclusive':''}`,String(bin.count),pct(bin.mean_confidence),pct(bin.accuracy)]) row.append(node('td',text));
    body.append(row);
  }
  table.append(body); const wrap=node('div',undefined,'table-wrap'); wrap.append(table); reliability.append(wrap);
  $('threshold').value=String(current.learned_policy.threshold);
  renderSandbox(current.learned_policy.threshold);
}
function renderAtlas() {
  const mode=$('atlas-filter').value, digit=$('atlas-digit').value, current=selectedAnalysis();
  const rows=current.rows.filter(row=>(digit==='all'||row.label===Number(digit)) && (mode==='all'||(mode==='candidate_errors'?row.candidate_prediction!==row.label:row.category===mode)));
  $('atlas-count').textContent=`${rows.length} matching cases · showing ${Math.min(atlasLimit,rows.length)} · ascending source row ID`;
  const grid=clear('atlas-cases'), examples=new Map(selectedRun().examples.map(e=>[e.row,e]));
  for(const row of rows.slice(0,atlasLimit)) {
    const example=examples.get(row.row), card=node('article',undefined,`case-card ${row.category}`);
    card.append(node('p',`ROW ${row.row} / ${row.category.replaceAll('_',' ')}`,'eyebrow'),digitImage(example),node('h4',`Actual ${row.label}`),node('p',`Linear ${row.baseline_prediction} → neural ${row.candidate_prediction}`),node('small',`${pct(row.candidate_confidence)} neural confidence`));
    grid.append(card);
  }
  if(!rows.length) grid.append(node('p','No cases match these filters.','fine'));
  $('atlas-more').hidden=rows.length<=atlasLimit;
}
function frontierAt(threshold) {
  // Upper-bound search over exact grouped-confidence operating points.
  const points=selectedAnalysis().frontier;
  let lo=0,hi=points.length;
  while(lo<hi) { const mid=(lo+hi)>>>1; if(points[mid].threshold<=threshold) lo=mid+1; else hi=mid; }
  return {...points[Math.max(0,lo-1)],threshold};
}
function svgNode(tag,attrs,text) {
  const element=document.createElementNS('http://www.w3.org/2000/svg',tag);
  for(const [key,value] of Object.entries(attrs)) element.setAttribute(key,String(value));
  if(text!==undefined) element.textContent=text;
  return element;
}
function renderSandbox(threshold) {
  const current=selectedAnalysis(), point=frontierAt(threshold), target=clear('sandbox-stats');
  $('threshold-value').textContent=`${pct(threshold)}${threshold===current.learned_policy.threshold?' · learned policy':' · exploratory'}`;
  target.append(stat('DEFERRED FOR REVIEW',`${point.deferred} / ${point.total}`,`${point.deferred_errors} error${point.deferred_errors===1?'':'s'} flagged; ${point.deferred_correct} correct answer${point.deferred_correct===1?'':'s'} deferred`),stat('ERRORS STILL ACCEPTED',String(point.accepted_errors),`Accepted accuracy: ${pct(point.accepted_accuracy)} · coverage ${pct(point.coverage)}`));
  $('sandbox-record').textContent=JSON.stringify({...point,scope:'Test-set exploration only; frozen policy and gate unchanged.'},null,2);
  const chart=svgNode('svg',{viewBox:'0 0 640 240',role:'img','aria-label':'Review workload versus errors still accepted. Exact operating points; not fitted or pooled.'});
  chart.append(svgNode('title',{},'Review workload versus accepted errors'));
  const maxErrors=Math.max(1,current.frontier[0].accepted_errors), x=p=>55+p.deferred/p.total*555, y=p=>175-p.accepted_errors/maxErrors*140;
  for(const fraction of [0,.25,.5,.75,1]) {
    const xpos=55+fraction*555;
    chart.append(svgNode('line',{x1:xpos,x2:xpos,y1:35,y2:175,class:'chart-grid'}),svgNode('text',{x:xpos,y:199,'text-anchor':'middle'},`${Math.round(fraction*100)}%`));
  }
  for(const count of [...new Set([0,Math.round(maxErrors/2),maxErrors])]) chart.append(svgNode('text',{x:40,y:180-count/maxErrors*140,'text-anchor':'end'},count));
  chart.append(svgNode('text',{x:55,y:20},'Errors still accepted'),svgNode('text',{x:330,y:228,'text-anchor':'middle'},'Predictions deferred for review'));
  chart.append(svgNode('polyline',{points:current.frontier.map(p=>`${x(p)},${y(p)}`).join(' '),class:'frontier-line'}));
  chart.append(svgNode('circle',{cx:x(current.learned_policy),cy:y(current.learned_policy),r:7,class:'learned-point'}));
  chart.append(svgNode('circle',{cx:x(point),cy:y(point),r:5,class:'selected-point'}));
  clear('frontier-chart').append(chart,node('p','○ Amber ring: learned policy · ● Mint point: exploratory threshold','fine'));
}
for(const id of ['atlas-filter','atlas-digit']) $(id).onchange=()=>{atlasLimit=12;renderAtlas();};
$('atlas-more').onclick=()=>{atlasLimit+=12;renderAtlas();};
$('threshold').oninput=()=>renderSandbox(Number($('threshold').value));
$('threshold-reset').onclick=()=>{$('threshold').value=String(selectedAnalysis().learned_policy.threshold);renderSandbox(selectedAnalysis().learned_policy.threshold);};
poll();
