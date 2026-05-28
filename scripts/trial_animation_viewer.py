import marimo

app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import json
    from gps_analysis import (
        build_trials, build_tracks_cache, build_gps_cache,
        build_arena_transforms, load_trial_tracks,
        detect_site_visits, detect_recruitment_episodes, SITE_GRID,
    )

    BAITED = {"A1", "A2", "A3"}

    print("Loading data...")
    _trials = build_trials()
    _gnss = build_gps_cache(_trials)
    _at = build_arena_transforms()
    TRACKS_CACHE = build_tracks_cache(_trials, gnss_cache=_gnss, arena_transforms=_at)

    test_trials = [t for t in _trials
                   if t["config"] in {"A", "B", "C", "D"}
                   and isinstance(t.get("assay"), int)
                   and t["date"] >= "2026-02-17"
                   and t["group_num"] not in {9, 14}
                   and t["group_size"] >= 2]

    trial_opts = {
        f"Assay {t['assay']} | Grp {t['group_num']} | {t['group_size']} sheep | Config {t['config']}": t
        for t in sorted(test_trials, key=lambda x: (x["assay"], x["group_num"]))
    }
    return mo, np, json, load_trial_tracks, detect_site_visits, detect_recruitment_episodes, BAITED, SITE_GRID, TRACKS_CACHE, trial_opts


@app.cell(hide_code=True)
def _(mo, trial_opts):
    mo.md("# Trial Animation Viewer")
    trial_dd = mo.ui.dropdown(options=trial_opts, label="Select trial")
    trial_dd
    return (trial_dd,)


@app.cell(hide_code=True)
def _(mo, np, json, load_trial_tracks, detect_site_visits, detect_recruitment_episodes, BAITED, SITE_GRID, TRACKS_CACHE, trial_dd, VIEWER_TPL):
    # ---- Compute data ----
    trial = trial_dd.value
    html_out = "<p><em>Select a trial above.</em></p>"

    if trial is not None:
        tracks = load_trial_tracks(trial, tracks_cache=TRACKS_CACHE, apply_orient=True)
        if tracks and len(tracks) >= 2:
            sids = sorted(tracks.keys())
            n_sheep = len(sids)
            max_t, sr = 35.0, 10
            pgx, pgy = [], []
            for sid in sids:
                trk = tracks[sid]
                o = np.argsort(trk["t"])
                tg = np.arange(0, max_t, 1.0 / (sr * 60))
                pgx.append(np.interp(tg, trk["t"][o], trk["gx"][o]) if len(trk["t"]) > 1 else np.full_like(tg, 2.5))
                pgy.append(np.interp(tg, trk["t"][o], trk["gy"][o]) if len(trk["t"]) > 1 else np.full_like(tg, 2.5))
            T = min(len(g) for g in pgx)
            GX = np.column_stack([g[:T] for g in pgx])
            GY = np.column_stack([g[:T] for g in pgy])
            cx, cy = GX.mean(axis=1), GY.mean(axis=1)
            # 1.5-second smoothing at 10 Hz = 15-sample kernel
            k = np.ones(15) / 15
            vx = np.convolve(np.gradient(cx), k, mode="same")
            vy = np.convolve(np.gradient(cy), k, mode="same")
            spd = np.sqrt(vx**2 + vy**2)
            vis = detect_site_visits(tracks, trial["field"], 0.5)
            disc = {s: float(min(v[1] for v in vl)) for s in BAITED if s in vis and vis[s] for vl in [vis[s]]}
            c10 = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
            sites = [{"label":l,"x":float(x),"y":float(y),"baited":l in BAITED} for l,(x,y) in SITE_GRID.items() if not l.startswith("E")]
            rl = lambda a: [round(float(v),3) for v in a]

            # Compute recruitment episodes using shared function
            sid_idx = {s: i for i, s in enumerate(sids)}
            raw_episodes = detect_recruitment_episodes(vis)
            episodes_json = [
                {"site": ep["site"], "time": round(ep["time"], 3),
                 "initiator": sid_idx.get(ep["initiator"], 0),
                 "followers": [{"idx": sid_idx[f["id"]], "time": round(f["time"], 3)}
                               for f in ep["followers"] if f["id"] in sid_idx]}
                for ep in raw_episodes
            ]

            dj = json.dumps({"trial_name":trial["name"],"assay":trial["assay"],"group_num":trial["group_num"],
                "config":trial["config"],"n_sheep":n_sheep,"sheep_ids":sids,"colors":[c10[i%10] for i in range(n_sheep)],
                "sample_rate":sr,"n_samples":T,"duration_s":T/sr,"sites":sites,"discoveries":disc,
                "episodes":episodes_json,
                "gx":[rl(GX[:,i]) for i in range(n_sheep)],"gy":[rl(GY[:,i]) for i in range(n_sheep)],
                "cx":rl(cx),"cy":rl(cy),"vx":rl(vx),"vy":rl(vy),"speed":rl(spd)}, separators=(",",":"))
            uid = f"av{abs(hash(trial['name']))%999999}"
            inner = VIEWER_TPL.replace("__UID__", uid).replace("__DATA__", dj)
            # Wrap in iframe so <script> executes (marimo strips scripts from mo.Html)
            import html as htmlmod
            escaped = htmlmod.escape(inner, quote=True)
            html_out = (
                f'<iframe srcdoc="{escaped}" '
                f'style="width:100%;max-width:920px;aspect-ratio:9/10;border:none;overflow:hidden;display:block;margin:0 auto" '
                f'sandbox="allow-scripts"></iframe>'
            )

    mo.output.append(mo.Html(html_out))


@app.cell(hide_code=True)
def _():
    VIEWER_TPL = '''<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:sans-serif;padding:8px}
.av-svg{width:100%;aspect-ratio:660/560;display:block}
</style></head><body>
<div id="__UID__" style="max-width:880px;margin:0 auto">
<div style="display:flex;align-items:center;gap:8px;padding:8px 0;flex-wrap:wrap">
  <button class="av-play" style="font-size:18px;width:40px;height:40px;border:1px solid #ccc;border-radius:6px;background:#fff;cursor:pointer">▶</button>
  <input class="av-scrub" type="range" min="0" max="1000" value="0" step="1" style="flex:1;min-width:200px;cursor:pointer">
  <span class="av-time" style="font-size:13px;font-weight:bold;min-width:100px">0:00.0</span>
  <label style="font-size:12px;display:flex;align-items:center;gap:4px">Speed:<input class="av-speed" type="range" min="1" max="50" value="10" style="width:80px"><span class="av-speed-lbl">10×</span></label>
  <label style="font-size:12px"><input class="av-lead" type="checkbox" checked> Leadership</label>
  <label style="font-size:12px"><input class="av-recruit" type="checkbox" checked> Recruitment</label>
  <label style="font-size:12px"><input class="av-trail" type="checkbox" checked> Trail</label>
</div>
<div class="av-status" style="font-size:12px;padding:2px 0;color:#666;height:1.4em;overflow:hidden">&nbsp;</div>
<svg class="av-svg" viewBox="-15 -15 660 560" style="width:100%;background:#f8f8f5;border:1px solid #ddd;border-radius:4px">
  <defs><marker id="ah-__UID__" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6 Z" fill="#222"/></marker></defs>
</svg>
</div>
<script>
(function(){
const root = document.getElementById("__UID__");
if (!root) return;
const svg = root.querySelector(".av-svg");
const ns = "http://www.w3.org/2000/svg";
const D = __DATA__;
const S = 100;
const SPD_THRESH = 0.000833;

function el(tag, attrs) {
  const e = document.createElementNS(ns, tag);
  for (const [k,v] of Object.entries(attrs)) e.setAttribute(k, String(v));
  return e;
}

svg.appendChild(el("rect",{x:0,y:0,width:500,height:500,fill:"none",stroke:"#aaa","stroke-width":"1.5"}));
for (let i=1;i<5;i++) {
  svg.appendChild(el("line",{x1:i*S,y1:0,x2:i*S,y2:500,stroke:"#eee","stroke-width":"0.5"}));
  svg.appendChild(el("line",{x1:0,y1:i*S,x2:500,y2:i*S,stroke:"#eee","stroke-width":"0.5"}));
}
for (let i=0;i<=5;i++) {
  const tx=el("text",{x:i*S,y:530,"text-anchor":"middle","font-size":"11",fill:"#888"});
  tx.textContent=i; svg.appendChild(tx);
  const ty=el("text",{x:-8,y:500-i*S+4,"text-anchor":"end","font-size":"11",fill:"#888"});
  ty.textContent=i; svg.appendChild(ty);
}

const stars={}, dzones={};
D.sites.forEach(s => {
  const sx=s.x*S, sy=(5-s.y)*S;
  svg.appendChild(el("circle",{cx:sx,cy:sy,r:18,fill:s.baited?"#E8B83D":"#e0e0e0",stroke:"#666","stroke-width":"0.8",opacity:"0.7"}));
  if (s.baited) {
    const dz=el("circle",{cx:sx,cy:sy,r:20,fill:"none",stroke:"#ccc","stroke-width":"1","stroke-dasharray":"4,3"});
    svg.appendChild(dz); dzones[s.label]=dz;
    const star=el("text",{x:sx,y:sy+7,"text-anchor":"middle","font-size":"30",fill:"#4AAD5B",display:"none"});
    star.textContent="★"; svg.appendChild(star); stars[s.label]=star;
  }
  const lb=el("text",{x:sx,y:sy+4,"text-anchor":"middle","font-size":"8","font-weight":"bold",fill:"#333"});
  lb.textContent=s.label; svg.appendChild(lb);
});

// Recruitment: site highlight rings (one per site label)
const siteCoords={};
D.sites.forEach(s=>{siteCoords[s.label]={x:s.x*S,y:(5-s.y)*S};});
const recruitRings={};
D.sites.forEach(s=>{
  if(s.label.startsWith("E"))return;
  const r=el("circle",{cx:s.x*S,cy:(5-s.y)*S,r:24,fill:"none","stroke-width":"3",opacity:"0.8",display:"none"});
  svg.appendChild(r); recruitRings[s.label]=r;
});

// Recruitment scoreboard background
const sbW=130, sbH=16+D.n_sheep*16, sbX=504, sbY=0;
const sbBg=el("rect",{x:sbX,y:sbY,width:sbW,height:sbH,fill:"white",stroke:"#ccc","stroke-width":"0.5",rx:"4",opacity:"0.9"});
svg.appendChild(sbBg);
const sbTitle=el("text",{x:sbX+sbW/2,y:sbY+12,"text-anchor":"middle","font-size":"9","font-weight":"bold",fill:"#555"});
sbTitle.textContent="Followers recruited"; svg.appendChild(sbTitle);
const sbRows=[];
for(let i=0;i<D.n_sheep;i++){
  const g=el("g",{});
  const dot=el("circle",{cx:sbX+10,cy:sbY+26+i*16,r:4,fill:D.colors[i]});
  g.appendChild(dot);
  const nm=el("text",{x:sbX+18,y:sbY+30+i*16,"font-size":"8",fill:"#333"});
  nm.textContent=D.sheep_ids[i]; g.appendChild(nm);
  const cnt=el("text",{x:sbX+sbW-8,y:sbY+30+i*16,"text-anchor":"end","font-size":"9","font-weight":"bold",fill:D.colors[i]});
  cnt.textContent="0"; g.appendChild(cnt);
  svg.appendChild(g);
  sbRows.push(cnt);
}

// Follower arrival arcs (small arcs near sites during active episodes)
const followerArcs=[];
for(let i=0;i<8;i++){
  const a=el("circle",{r:5,fill:"none","stroke-width":"2",opacity:"0",display:"none"});
  svg.appendChild(a); followerArcs.push(a);
}

const trailEls=[], projEls=[];
for (let i=0;i<D.n_sheep;i++) {
  const pl=el("polyline",{fill:"none",stroke:D.colors[i],"stroke-width":"1.5",opacity:"0.35","stroke-linejoin":"round","stroke-linecap":"round"});
  svg.appendChild(pl); trailEls.push(pl);
  const ln=el("line",{stroke:D.colors[i],"stroke-width":"1.2","stroke-dasharray":"3,2",opacity:"0.5",display:"none"});
  svg.appendChild(ln); projEls.push(ln);
}

const projAxis=el("line",{stroke:"#ddd","stroke-width":"1",display:"none"});
svg.appendChild(projAxis);
const velLine=el("line",{stroke:"#222","stroke-width":"2.5","marker-end":"url(#ah-__UID__)",display:"none"});
svg.appendChild(velLine);

const centG=el("g",{});
centG.appendChild(el("line",{x1:-7,y1:0,x2:7,y2:0,stroke:"#000","stroke-width":"2.5"}));
centG.appendChild(el("line",{x1:0,y1:-7,x2:0,y2:7,stroke:"#000","stroke-width":"2.5"}));
svg.appendChild(centG);

const leaderRing=el("circle",{r:14,fill:"none",stroke:"#E8B83D","stroke-width":"3",display:"none"});
svg.appendChild(leaderRing);

const sheepEls=[], sheepLbls=[];
for (let i=0;i<D.n_sheep;i++) {
  const c=el("circle",{r:8,fill:D.colors[i],stroke:"#333","stroke-width":"0.8"});
  svg.appendChild(c); sheepEls.push(c);
  const lb=el("text",{"font-size":"7","font-weight":"bold",fill:D.colors[i]});
  lb.textContent=D.sheep_ids[i]; svg.appendChild(lb); sheepLbls.push(lb);
}

let playing=false, tSec=0, lastTs=null, speed=10;
const maxT=D.duration_s;
const trailBufs=Array.from({length:D.n_sheep},()=>[]);
const TRAIL_PTS=30;

const playBtn=root.querySelector(".av-play");
const scrub=root.querySelector(".av-scrub");
const timeDisp=root.querySelector(".av-time");
const speedSl=root.querySelector(".av-speed");
const speedLbl=root.querySelector(".av-speed-lbl");
const leadCk=root.querySelector(".av-lead");
const recruitCk=root.querySelector(".av-recruit");
const trailCk=root.querySelector(".av-trail");
const statusDiv=root.querySelector(".av-status");

scrub.max=Math.floor(maxT*10);

playBtn.onclick=()=>{
  playing=!playing;
  playBtn.textContent=playing?"⏸":"▶";
  lastTs=null;
  if(playing&&tSec>=maxT)tSec=0;
};
scrub.oninput=()=>{
  tSec=parseFloat(scrub.value)/10;
  trailBufs.forEach(b=>b.length=0);
  render(tSec);
};
speedSl.oninput=()=>{
  speed=parseInt(speedSl.value);
  speedLbl.textContent=speed+"×";
};

function fmt(s){const m=Math.floor(s/60),sec=(s%60).toFixed(1);return m+":"+(sec<10?"0":"")+sec;}

function render(t){
  const idx=Math.min(Math.floor(t*D.sample_rate),D.n_samples-1);
  if(idx<0)return;
  const showLead=leadCk.checked, showTrail=trailCk.checked;

  for(let i=0;i<D.n_sheep;i++){
    const gx=D.gx[i][idx],gy=D.gy[i][idx];
    const sx=gx*S,sy=(5-gy)*S;
    sheepEls[i].setAttribute("cx",sx);
    sheepEls[i].setAttribute("cy",sy);
    sheepLbls[i].setAttribute("x",sx+10);
    sheepLbls[i].setAttribute("y",sy-8);
    if(showTrail){
      trailBufs[i].push(sx+","+sy);
      if(trailBufs[i].length>TRAIL_PTS)trailBufs[i].shift();
      trailEls[i].setAttribute("points",trailBufs[i].join(" "));
      trailEls[i].setAttribute("display","");
    }else{trailEls[i].setAttribute("display","none");}
  }

  const cxv=D.cx[idx]*S,cyv=(5-D.cy[idx])*S;
  centG.setAttribute("transform","translate("+cxv+","+cyv+")");

  const spd=D.speed[idx],moving=spd>SPD_THRESH;

  if(showLead&&moving){
    const vx=D.vx[idx],vy=D.vy[idx],vnx=vx/spd,vny=vy/spd;
    const aS=500;
    velLine.setAttribute("x1",cxv);velLine.setAttribute("y1",cyv);
    velLine.setAttribute("x2",cxv+vx*aS);velLine.setAttribute("y2",cyv-vy*aS);
    velLine.setAttribute("display","");
    projAxis.setAttribute("x1",cxv-vnx*200);projAxis.setAttribute("y1",cyv+vny*200);
    projAxis.setAttribute("x2",cxv+vnx*250);projAxis.setAttribute("y2",cyv-vny*250);
    projAxis.setAttribute("display","");
    let maxP=-Infinity,lIdx=0;
    for(let i=0;i<D.n_sheep;i++){
      const dx=D.gx[i][idx]-D.cx[idx],dy=D.gy[i][idx]-D.cy[idx];
      const p=dx*vnx+dy*vny;
      if(p>maxP){maxP=p;lIdx=i;}
      const px=(D.cx[idx]+p*vnx)*S,py=(5-(D.cy[idx]+p*vny))*S;
      const sx2=D.gx[i][idx]*S,sy2=(5-D.gy[i][idx])*S;
      projEls[i].setAttribute("x1",sx2);projEls[i].setAttribute("y1",sy2);
      projEls[i].setAttribute("x2",px);projEls[i].setAttribute("y2",py);
      projEls[i].setAttribute("display","");
    }
    const lx=D.gx[lIdx][idx]*S,ly=(5-D.gy[lIdx][idx])*S;
    leaderRing.setAttribute("cx",lx);leaderRing.setAttribute("cy",ly);
    leaderRing.setAttribute("display","");
    statusDiv.textContent="Leader: "+D.sheep_ids[lIdx]+" \u2022 Speed: "+(spd*D.sample_rate*60*10).toFixed(1)+" m/min";
  }else{
    velLine.setAttribute("display","none");
    projAxis.setAttribute("display","none");
    leaderRing.setAttribute("display","none");
    projEls.forEach(l=>l.setAttribute("display","none"));
    statusDiv.innerHTML=moving?"\u00a0":"\u00a0";
  }

  for(const[site,dt] of Object.entries(D.discoveries)){
    if(stars[site])stars[site].setAttribute("display",t>=dt*60?"":"none");
    if(dzones[site]){
      dzones[site].setAttribute("stroke",t>=dt*60?"#4AAD5B":"#ccc");
      dzones[site].setAttribute("stroke-width",t>=dt*60?"2":"1");
    }
  }

  // ---- Recruitment episodes ----
  const showRecruit=recruitCk.checked;
  const tMin=t/60;
  const recruitCounts=new Array(D.n_sheep).fill(0);
  // Hide all rings and arcs first
  Object.values(recruitRings).forEach(r=>r.setAttribute("display","none"));
  followerArcs.forEach(a=>{a.setAttribute("display","none");a.setAttribute("opacity","0");});

  if(showRecruit && D.episodes){
    let arcIdx=0;
    for(const ep of D.episodes){
      if(tMin<ep.time) continue;
      // Count followers attracted by this initiator (only those arrived so far)
      const nArrived=ep.followers.filter(f=>tMin>=f.time).length;
      recruitCounts[ep.initiator]+=nArrived;
      const sc=siteCoords[ep.site];
      if(!sc) continue;
      // Active episode: within 2 min of start and has followers still arriving
      const epEnd=ep.followers.length>0?Math.max(...ep.followers.map(f=>f.time)):ep.time;
      const isActive=tMin>=ep.time && tMin<=epEnd+0.1;
      if(isActive){
        // Highlight site with initiator color
        const ring=recruitRings[ep.site];
        if(ring){
          ring.setAttribute("stroke",D.colors[ep.initiator]);
          ring.setAttribute("display","");
        }
        // Show follower arrival arcs
        for(const f of ep.followers){
          if(tMin>=f.time && arcIdx<followerArcs.length){
            const age=tMin-f.time;
            const fade=Math.max(0,1-age/0.2);
            const a=followerArcs[arcIdx];
            const angle=(arcIdx*Math.PI*2)/Math.max(ep.followers.length,1);
            a.setAttribute("cx",sc.x+Math.cos(angle)*28);
            a.setAttribute("cy",sc.y+Math.sin(angle)*28);
            a.setAttribute("stroke",D.colors[f.idx]);
            a.setAttribute("opacity",String(fade));
            a.setAttribute("display","");
            arcIdx++;
          }
        }
      }
    }
    // Update scoreboard
    for(let i=0;i<D.n_sheep;i++) sbRows[i].textContent=String(recruitCounts[i]);
    sbBg.setAttribute("display","");
    sbTitle.setAttribute("display","");
  }else{
    sbBg.setAttribute("display",showRecruit?"":"none");
    sbTitle.setAttribute("display",showRecruit?"":"none");
    for(let i=0;i<D.n_sheep;i++) sbRows[i].parentNode.style.display=showRecruit?"":"none";
  }

  const nF=Object.values(D.discoveries).filter(dt=>t>=dt*60).length;
  const comp=nF===3;
  timeDisp.textContent=fmt(t)+(comp?" ✓ COMPLETE":" | "+nF+"/3");
  timeDisp.style.color=comp?"#4AAD5B":"#333";
  scrub.value=Math.floor(t*10);
}

function animate(ts){
  if(playing){
    if(lastTs!==null){
      const dt=(ts-lastTs)/1000;
      tSec=Math.min(tSec+dt*speed,maxT);
      if(tSec>=maxT){playing=false;playBtn.textContent="▶";}
    }
    lastTs=ts;
  }else{lastTs=null;}
  render(tSec);
  requestAnimationFrame(animate);
}

render(0);
requestAnimationFrame(animate);
})();
</script>
</body></html>'''
    return (VIEWER_TPL,)


if __name__ == "__main__":
    app.run()
