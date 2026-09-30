// Экономический атлас: одна страница, данные из data/atlas.js (собирает scripts/build_site.py).
(() => {
  const A = window.ATLAS;
  if (!A || !A.meta) { document.body.insertAdjacentHTML("afterbegin", "<p style='padding:24px'>Нет data/atlas.js — запустите scripts/build_site.py.</p>"); return; }
  const M = A.meta, T = M.types, K = T.length, CATS = M.categories, NM = M.months.length;
  const ru = d3.formatLocale({ decimal: ",", thousands: " ", grouping: [3], currency: ["", " ₽"] });
  const pct = ru.format(".1%"), int = ru.format(",d"), f2 = ru.format(".2f"), f3 = ru.format(".3f");
  const P = x => pct(x).replace("%", " %"), P0 = x => ru.format(".0%")(x).replace("%", " %");
  const sgn = x => (x > 0 ? "+" : x < 0 ? "−" : "") + P(Math.abs(x));
  const xr = r => "×" + ru.format(".2f")(r);
  const plural = (n, one, few, many) => { const a = n % 10, b = n % 100; return a === 1 && b !== 11 ? one : a >= 2 && a <= 4 && (b < 12 || b > 14) ? few : many; };
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const mName = i => { const [y, m] = M.months[i].split("-"); return `${MONTHS[+m - 1]} ${y}`; };
  const CAT_S = { "Продовольствие": "Продукты", "Маркетплейсы": "Маркетплейсы", "Общественное питание": "Кафе и рестораны", "Здоровье": "Здоровье", "Транспорт": "Транспорт" };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const $ = s => document.querySelector(s);
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const tip = $("#tip");
  const showTip = (e, html) => {
    tip.innerHTML = html; tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + w > innerWidth - 8) x = e.clientX - w - 14;
    if (y + h > innerHeight - 8) y = e.clientY - h - 14;
    tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
  };
  const hideTip = () => { tip.hidden = true; };

  const L = A.labels, S = A.shares, sc = S.scale, LY = A.layout, geo = A.geo, N = L.ids.length;
  const row = new Map(L.ids.map((id, i) => [id, i]));
  const feat = new Map(geo.features.map(f => [f.id, f]));
  // Названия из types.csv длинные («Москва, Подмосковье…: самые высокие траты…»): коротко — до двоеточия.
  const short = n => n.split(":")[0].trim(), rest = n => n.includes(":") ? n.slice(n.indexOf(":") + 1).trim() : "";
  const tColor = k => (k >= 0 && T[k] ? T[k].color : "#d9d6cf");
  const ruMonth = d3.range(NM).map(t => CATS.map((_, c) => d3.mean(S.values, v => v[t][c]) / sc));
  // Основной тип МО — самый частый за 24 месяца (при равенстве — старший номер, как в build_site.py).
  const modal = L.labels.map(r => { const c = new Array(K).fill(0); r.forEach(k => k >= 0 && c[k]++); const mx = Math.max(...c); return c.lastIndexOf(mx); });
  const count = d3.range(NM).map(m => T.map((_, k) => L.labels.reduce((a, r) => a + (r[m] === k), 0)));
  const bySpend = d3.range(K).sort((a, b) => (T[b].spend_ratio ?? 0) - (T[a].spend_ratio ?? 0));
  const gt = T.map(t => t.growth_total ?? 0), hiG = d3.maxIndex(gt), loG = d3.minIndex(gt);
  const kTop = bySpend[0], kBot = bySpend[K - 1];

  // ---- Шапка
  const namesStub = (M.stub || []).includes("названия типов");
  if (namesStub) $("#types .sec-head").insertAdjacentHTML("afterend", `<p class="stub-note"><b>Черновые названия.</b> «Тип A…» — временные метки. Состав типов и все числа — настоящие.</p>`);
  $("#deck").innerHTML = `Мы разложили ${int(M.n_mo)} муниципалитетов России на ${K} типов местной экономики — по тому, на что жители тратят деньги с карты, — и проследили каждый из ${NM} месяцев 2023–2024 годов. `
    + `В типе «${esc(short(T[hiG].name))}» траты за 2024 год выросли на <b>${P(gt[hiG])}</b>, в типе «${esc(short(T[loG].name))}» — на <b>${P(gt[loG])}</b>: разрыв между городом и селом растёт.`;
  const spendGap = T[kTop].spend_ratio / T[kBot].spend_ratio;
  $("#keynums").innerHTML = [
    [K, `типов местной экономики у&nbsp;${int(M.n_mo)} МО из ${M.n_regions} регионов`],
    [xr(spendGap).replace(/0$/, ""), `разница в тратах на жителя между типами «${esc(short(T[kTop].name).split(",")[0])}» и&nbsp;«${esc(short(T[kBot].name).replace("Сельские районы с самыми низкими тратами", "сельские районы"))}»`],
    [P0(d3.mean(L.stability)), `месяцев муниципалитет проводит в&nbsp;своём основном типе`],
  ].map(([v, k]) => `<div><dd>${v}</dd><dt>${k}</dt></div>`).join("");
  $("#headline").textContent = M.headline;
  $("#built").textContent = M.built;
  const lk = M.links, link = (u, t) => u && u !== "#" ? `<a href="${esc(u)}">${t}</a>` : `<span title="Появится к сдаче">${t} — скоро</span>`;
  $("#links").innerHTML = link(lk.repo, "Репозиторий с кодом") + link(lk.report, "Отчёт") + link(lk.pdf, "PDF-версия");

  // Картограммы из build_site.py: км той же проекции, что geo.json (ось y вверх).
  const eqXY = d3.range(N).map(i => [LY.eq[0][i], LY.eq[1][i]]), popXY = d3.range(N).map(i => [LY.pop[0][i], LY.pop[1][i]]);
  const cenXY = d3.range(N).map(i => [LY.c[0][i], LY.c[1][i]]);
  const HEXR = LY.r_eq * 2 / Math.sqrt(3);  // шестиугольник решётки с шагом 2r
  const fitter = (pts, rad, W, H, pad) => {
    const x0 = d3.min(pts, (p, i) => p[0] - rad(i)), x1 = d3.max(pts, (p, i) => p[0] + rad(i));
    const y0 = d3.min(pts, (p, i) => p[1] - rad(i)), y1 = d3.max(pts, (p, i) => p[1] + rad(i));
    const k = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    const ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
    const f = p => [ox + (p[0] - x0) * k, oy + (y1 - p[1]) * k];
    f.k = k; f.aspect = (x1 - x0) / (y1 - y0); return f;
  };
  const PLACES = [["Москва", "Москва"], ["Санкт-Петербург", "Петербург"], ["Республика Саха (Якутия)", "Якутия"], ["Свердловская область", "Свердловская обл."],
    ["Алтайский край", "Алтайский край"], ["Приморский край", "Приморье"], ["Красноярский край", "Красноярский край"], ["Саратовская область", "Саратовская обл."], ["Республика Башкортостан", "Башкирия"]]
    .map(([r, label]) => ({ label, idx: d3.range(N).filter(i => feat.get(L.ids[i]).properties.r === r) })).filter(p => p.idx.length);

  const HERO_PLACES = ["Москва", "Петербург", "Якутия", "Приморье", "Алтайский край"];
  // ================= Первый экран: 2 016 точек перетекают между типами =================
  (function hero() {
    const stage = $(".hv-stage"), cv = $("#hv-canvas"), ctx = cv.getContext("2d"), labs = $("#hv-labels");
    const X = new Float32Array(N), Y = new Float32Array(N), R = new Float32Array(N);
    const X0 = new Float32Array(N), Y0 = new Float32Array(N), R0 = new Float32Array(N);
    const X1 = new Float32Array(N), Y1 = new Float32Array(N), R1 = new Float32Array(N), DL = new Float32Array(N);
    const hi = bySpend.filter(k => (T[k].spend_ratio ?? 1) >= 1), lo = bySpend.filter(k => (T[k].spend_ratio ?? 1) < 1);
    const maxN = d3.max(count.flat());
    let W = 0, H = 0, Hmap = 0, Htypes = 0, dot = 2.5, cen = [], rows = [], mapF = null, mode = "map", month = 0, hover = -1;
    let anim = null, playing = !reduce, visible = true, timer = null, userPaused = false;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const colorOf = i => T[modal[i]].color;
    const groups = T.map((_, k) => d3.range(N).filter(i => modal[i] === k));

    function layout() {
      W = stage.clientWidth || 700;
      const s = W >= 860 ? 6.2 : W >= 600 ? 5.4 : 4.3;
      dot = s * 0.4;
      const c = s * 0.53, Rm = c * Math.sqrt(maxN) + dot, lab = W >= 600 ? 96 : 104, head = 34;
      const cols = W >= 600 ? 4 : 2;
      cen = []; rows = []; let y = 0;
      [[hi, "Траты на жителя выше средних по стране"], [lo, "Траты на жителя ниже средних"]].forEach(([grp, title]) => {
        rows.push({ y, title }); y += head;
        for (let a = 0; a < grp.length; a += cols) {
          const ch = grp.slice(a, a + cols), cw = W / Math.max(ch.length, cols === 4 ? 3.2 : 2);
          const x0 = (W - cw * ch.length) / 2;
          ch.forEach((k, j) => { cen[k] = { x: x0 + cw * (j + 0.5), y: y + Rm, c, ly: y + 2 * Rm + 10, w: cw - 14 }; });
          y += 2 * Rm + lab;
        }
      });
      Htypes = Math.round(y);
      const aspect = fitter(eqXY, () => LY.r_eq, 100, 100, 0).aspect;
      Hmap = Math.round(Math.min(W / aspect, Math.max(Htypes, 380)));
      if (W >= 600) Hmap = Math.max(Hmap, Htypes);  // на широком экране высота не прыгает при смене расклада
      mapF = fitter(eqXY, () => LY.r_eq, W, Hmap, 4);
      labs.innerHTML = rows.map(r => `<div class="hv-row" style="top:${r.y}px">${r.title}</div>`).join("")
        + bySpend.map(k => `<div class="hv-l" data-k="${k}" style="--c:${T[k].color};left:${cen[k].x}px;top:${cen[k].ly}px;--w:${cen[k].w}px">
            <div class="n"></div><div class="t">${esc(short(T[k].name))}</div><div class="s">траты ${xr(T[k].spend_ratio)} к среднему</div></div>`).join("")
        + PLACES.filter(p => HERO_PLACES.includes(p.label)).map(p => { const q = d3.range(p.idx.length).map(j => mapF(eqXY[p.idx[j]])); return `<div class="hv-place" style="left:${d3.mean(q, d => d[0])}px;top:${d3.mean(q, d => d[1])}px">${p.label}</div>`; }).join("");
      updateLabels();
    }
    const size = h => { H = h; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); cv.style.height = H + "px"; stage.style.height = H + "px"; };

    function targets() {
      if (mode === "map") {
        const r = HEXR * mapF.k * 0.9;
        for (let i = 0; i < N; i++) { const p = mapF(eqXY[i]); X1[i] = p[0]; Y1[i] = p[1]; R1[i] = r; }
        return;
      }
      for (let k = 0; k < K; k++) {
        const mem = [];
        for (let i = 0; i < N; i++) if (L.labels[i][month] === k) mem.push(i);
        // в центре — «свои» МО (основной тип совпадает), по убыванию устойчивости; «гости» — по краю
        mem.sort((a, b) => (modal[b] === k) - (modal[a] === k) || L.stability[b] - L.stability[a] || a - b);
        const { x, y, c } = cen[k];
        mem.forEach((i, j) => { const th = j * 2.39996323, rr = c * Math.sqrt(j + 0.5); X1[i] = x + rr * Math.cos(th); Y1[i] = y + rr * Math.sin(th); R1[i] = dot; });
      }
      for (let i = 0; i < N; i++) if (L.labels[i][month] < 0) { X1[i] = -20; Y1[i] = -20; R1[i] = 0; }
    }
    function go(dur, delayOf) {
      X0.set(X); Y0.set(Y); R0.set(R);
      for (let i = 0; i < N; i++) DL[i] = delayOf ? delayOf(i) : 0;
      if (reduce || !dur) { X.set(X1); Y.set(Y1); R.set(R1); anim = null; draw(); return; }
      anim = { t0: performance.now(), dur, end: dur + d3.max(DL) };
      requestAnimationFrame(frame);
    }
    function frame(now) {
      if (!anim) return;
      const t = now - anim.t0;
      for (let i = 0; i < N; i++) {
        const e = d3.easeCubicInOut(Math.max(0, Math.min(1, (t - DL[i]) / anim.dur)));
        X[i] = X0[i] + (X1[i] - X0[i]) * e; Y[i] = Y0[i] + (Y1[i] - Y0[i]) * e; R[i] = R0[i] + (R1[i] - R0[i]) * e;
      }
      draw();
      if (t < anim.end) requestAnimationFrame(frame); else { anim = null; if (mode === "map" && H !== Hmap) { size(Hmap); draw(); } }
    }
    function draw() {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      for (let k = 0; k < K; k++) {
        ctx.beginPath(); ctx.fillStyle = T[k].color;
        for (const i of groups[k]) { if (R[i] <= 0.05) continue; ctx.moveTo(X[i] + R[i], Y[i]); ctx.arc(X[i], Y[i], R[i], 0, 6.2832); }
        ctx.fill();
      }
      if (hover >= 0) { ctx.beginPath(); ctx.arc(X[hover], Y[hover], R[hover] + 3, 0, 6.2832); ctx.lineWidth = 1.5; ctx.strokeStyle = "#121317"; ctx.stroke(); }
    }
    function updateLabels() {
      labs.querySelectorAll(".hv-l").forEach(el => { const k = +el.dataset.k; el.querySelector(".n").innerHTML = `${int(count[month][k])}<small>МО</small>`; });
      stage.classList.toggle("hv-off", mode === "map"); stage.classList.toggle("hv-on", mode !== "map");
      $("#hv-month").textContent = mode === "map" ? "Основной тип за 2023–2024" : mName(month);
      $("#hv-ticks").innerHTML = d3.range(NM).map(m => `<i class="${mode !== "map" && m <= month ? "on" : ""}${m === 12 ? " y" : ""}"></i>`).join("");
      const key = bySpend.map(k => `<i style="background:${T[k].color}"></i>${esc(short(T[k].name))}`).join(" · ");
      $("#hv-cap").innerHTML = mode === "map"
        ? `Каждая точка — муниципалитет, все одного размера: Москва из ${PLACES[0] ? PLACES[0].idx.length : 144} районов весит здесь больше, чем Якутия. Цвет — основной тип за 24 месяца: ${key}.`
        : `Каждая точка — муниципалитет. Группа — его тип в этом месяце, цвет — основной тип за два года: точка чужого цвета на краю группы — МО «в гостях». За месяц тип меняют ${int(month ? L.labels.reduce((a, r) => a + (r[month] !== r[month - 1]), 0) : 0)} МО.`;
    }
    function setMode(m, dur = 1300) {
      mode = m;
      document.querySelectorAll("[data-hv]").forEach(b => b.classList.toggle("on", b.dataset.hv === m));
      if (m !== "map" || H < Hmap) size(Math.max(Htypes, Hmap));
      targets(); updateLabels();
      go(dur, i => m === "map" ? (mapF(eqXY[i])[0] / W) * 500 : ((X[i] / W) * 450 + (i % 7) * 20));
    }
    function setMonth(m) {
      const prev = month; month = m;
      const moved = i => L.labels[i][m] !== L.labels[i][prev];
      targets(); updateLabels();
      go(1000, i => moved(i) ? 150 + (i * 37 % 350) : 0);
    }
    const tick = () => {
      if (!playing || !visible) return;
      if (mode === "map") setMode("types");
      else if (month === NM - 1) { setMonth(0); }
      else setMonth(month + 1);
      timer = setTimeout(tick, month === NM - 1 ? 3600 : 1700);
    };
    const schedule = d => { clearTimeout(timer); timer = setTimeout(tick, d); };
    const setPlay = p => { playing = p; $("#hv-play").textContent = p ? "❚❚" : "▶"; $("#hv-play").setAttribute("aria-label", p ? "Пауза" : "Проиграть"); if (p) schedule(400); else clearTimeout(timer); };
    $("#hv-play").onclick = () => { userPaused = playing; setPlay(!playing); };
    // «На карте» — выбор читателя: месяцы там не листаются (цвет — основной тип), поэтому пауза; «По типам» — снова играет
    document.querySelectorAll("[data-hv]").forEach(b => b.onclick = () => {
      if (b.dataset.hv === mode) return;
      setMode(b.dataset.hv);
      if (b.dataset.hv === "map") setPlay(false); else if (!userPaused) { setPlay(true); schedule(2200); }
    });
    new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible && playing) schedule(600); }, { threshold: 0.15 }).observe(stage);

    cv.addEventListener("pointermove", e => {
      const b = cv.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top;
      let best = -1, bd = 64;
      for (let i = 0; i < N; i++) { const d = (X[i] - x) ** 2 + (Y[i] - y) ** 2; if (d < bd) { bd = d; best = i; } }
      if (best !== hover) { hover = best; if (!anim) draw(); }
      if (best < 0) return hideTip();
      const f = feat.get(L.ids[best]).properties, k = L.labels[best][month], mk = modal[best];
      showTip(e, `<b>${esc(f.n)}</b><div class="r">${esc(f.r)}</div>` + (mode === "map" ? "" : `<span class="type-pill"><i style="background:${tColor(k)}"></i>${esc(short(T[k].name))}</span><div class="r">тип в месяце «${mName(month)}»</div>`)
        + `<span class="type-pill"><i style="background:${tColor(mk)}"></i>${esc(short(T[mk].name))}</span><div class="r">основной тип: ${P0(L.stability[best])} месяцев</div>`);
    });
    cv.addEventListener("pointerleave", () => { hover = -1; hideTip(); if (!anim) draw(); });

    let rw = 0;
    const init = () => {
      layout(); size(reduce ? Math.max(Htypes, Hmap) : Hmap);
      mode = reduce ? "types" : "map";
      document.querySelectorAll("[data-hv]").forEach(b => b.classList.toggle("on", b.dataset.hv === mode));
      targets(); X.set(X1); Y.set(Y1); R.set(R1); updateLabels(); draw();
    };
    init();
    if (playing) schedule(2400); else setPlay(false);
    addEventListener("resize", () => { if (stage.clientWidth === rw) return; rw = stage.clientWidth; clearTimeout(init.t); init.t = setTimeout(() => {
      layout(); size(mode === "map" ? Hmap : Math.max(Htypes, Hmap)); targets(); X.set(X1); Y.set(Y1); R.set(R1); updateLabels(); draw(); }, 150); });
    rw = stage.clientWidth;
  })();

  // ================= 01. Карта: территория / равные клетки / по населению =================
  (function map() {
    const svg = d3.select("#map-svg"), W = 1000, H = 556;
    svg.attr("viewBox", `0 0 ${W} ${H}`);
    const g = svg.append("g");
    const proj = d3.geoIdentity().reflectY(true).fitExtent([[12, 12], [W - 12, H - 12]], geo);
    const path = d3.geoPath(proj);
    const fEq = fitter(eqXY, () => LY.r_eq, W, H, 14), fPop = fitter(popXY, i => LY.r_pop[i], W, H, 14);
    let month = 0, layer = "type", isolate = null, pinned = null, form = "area";
    const stabColor = d3.scaleSequential([0.25, 1], d3.interpolateRgbBasis(["#d6e6f7", "#7fb0e6", "#2d6fbe", "#0d3163"])).clamp(true);
    const f2016 = L.ids.map(id => feat.get(id));
    const paths = g.append("g").attr("class", "geo").selectAll("path").data(geo.features).join("path").attr("d", path);
    const hex = d3.range(6).map(a => [Math.sin(a * Math.PI / 3), -Math.cos(a * Math.PI / 3)]);
    const HEX = "M" + hex.map(p => p.map(v => (v * 0.97).toFixed(3)).join(",")).join("L") + "Z", CIRC = "M1,0A1,1 0 1,1 -1,0A1,1 0 1,1 1,0Z";
    const tiles = g.append("g").attr("class", "tiles").style("display", "none").selectAll("path").data(f2016).join("path");
    const placesG = g.append("g");
    const fill = d => {
      const i = row.get(d.id); if (i === undefined) return "#e4e1da";
      if (layer === "stab") return stabColor(L.stability[i]);
      const k = L.labels[i][month];
      return isolate === null || isolate === k ? tColor(k) : "#e7e4dd";
    };
    const paint = () => { paths.attr("fill", fill); tiles.attr("fill", fill); };
    const pos = (i, f = form) => f === "eq" ? [...fEq(eqXY[i]), HEXR * fEq.k] : f === "pop" ? [...fPop(popXY[i]), LY.r_pop[i] * fPop.k] : [...proj(cenXY[i]), 0.01];
    const tf = (i, f) => { const [x, y, r] = pos(i, f); return `translate(${x.toFixed(1)},${y.toFixed(1)}) scale(${Math.max(0.01, r).toFixed(3)})`; };

    // Подписи мест — ориентиры, без них картограмму не узнать
    const places = placesG.selectAll("text").data(PLACES).join("text").attr("class", "place").attr("text-anchor", "middle").text(p => p.label);
    let zk = 1;
    const placeXY = (p, f) => { const q = p.idx.map(i => pos(i, f)); return [d3.mean(q, d => d[0]), d3.mean(q, d => d[1])]; };
    const fontFix = () => { const w = svg.node().clientWidth || W; places.style("font-size", (12 * W / w / zk) + "px").style("display", p => w < 600 && !HERO_PLACES.includes(p.label) ? "none" : null).style("stroke-width", 3 * W / w / zk); };

    function setForm(f, animate = true) {
      const from = form; form = f;
      document.querySelectorAll("[data-geo]").forEach(b => b.classList.toggle("on", b.dataset.geo === f));
      const dur = animate && !reduce ? 1100 : 0, delay = (_, i) => dur ? pos(i, f === "area" ? from : f)[0] / W * 400 : 0;
      if (f !== "area") {
        tiles.attr("d", f === "eq" ? HEX : CIRC);
        if (from === "area") tiles.attr("transform", (_, i) => tf(i, "area"));
        d3.select(".tiles").style("display", null);
        tiles.transition().duration(dur).delay(delay).ease(d3.easeCubicInOut).attr("transform", (_, i) => tf(i, f));
        paths.transition().duration(dur * 0.6).style("opacity", 0).on("end", function () { d3.select(this).style("visibility", "hidden"); });
      } else {
        paths.style("visibility", null).transition().duration(dur * 0.8).delay(dur * 0.3).style("opacity", 1);
        tiles.transition().duration(dur).delay(delay).ease(d3.easeCubicInOut).attr("transform", (_, i) => tf(i, "area"))
          .end().then(() => form === "area" && d3.select(".tiles").style("display", "none")).catch(() => {});
        if (!dur) d3.select(".tiles").style("display", "none");
      }
      places.transition().duration(dur).attr("x", p => placeXY(p, f)[0]).attr("y", p => placeXY(p, f)[1] - (f === "area" ? 8 : 0));
      $("#map-hint").textContent = f === "eq" ? "Каждое МО — одна клетка одного размера" : f === "pop" ? "Площадь кружка — население МО, 2024" : "Колесо с Ctrl или два пальца — масштаб";
    }

    const zoom = d3.zoom().scaleExtent([1, 40]).translateExtent([[0, 0], [W, H]])
      .filter(e => e.type === "wheel" ? (e.ctrlKey || e.metaKey) : e.type.startsWith("touch") ? e.touches.length > 1 : !e.button)
      .on("zoom", e => { g.attr("transform", e.transform); zk = e.transform.k; fontFix(); });
    svg.call(zoom).style("touch-action", "pan-y");
    document.querySelectorAll(".zoom button").forEach(b => b.onclick = () => {
      const z = b.dataset.z;
      if (z === "reset") svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
      else svg.transition().duration(300).call(zoom.scaleBy, z === "in" ? 2 : 0.5);
    });
    const zoomTo = d => {
      let x0, y0, x1, y1;
      if (form === "area") [[x0, y0], [x1, y1]] = path.bounds(d);
      else { const [x, y, r] = pos(row.get(d.id)); [x0, y0, x1, y1] = [x - r * 6, y - r * 6, x + r * 6, y + r * 6]; }
      const k = Math.min(20, 0.35 / Math.max((x1 - x0) / W, (y1 - y0) / H));
      svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
    };

    const sharesHtml = i => { const v = S.values[i][month]; return "<table>" + CATS.map((c, j) => `<tr><td>${CAT_S[c]}</td><td>${P(v[j] / sc)}</td></tr>`).join("") + "</table>"; };
    const sel = () => { paths.classed("sel", p => p === pinned); tiles.classed("sel", p => p === pinned); if (pinned) { paths.filter(p => p === pinned).raise(); tiles.filter(p => p === pinned).raise(); } };
    [paths, tiles].forEach(s => s.on("mousemove", (e, d) => {
      const i = row.get(d.id);
      if (i === undefined) return showTip(e, `<b>${esc(d.properties.n)}</b><div class="r">${esc(d.properties.r)}</div>Нет полных 24 месяцев данных — в типологию не входит.`);
      const k = L.labels[i][month], pop = LY.population[i];
      showTip(e, `<b>${esc(d.properties.n)}</b><div class="r">${esc(d.properties.r)}${pop ? ` · ${int(pop)} жителей` : ""}</div>
        <span class="type-pill"><i style="background:${tColor(k)}"></i>${k >= 0 ? esc(T[k].name) : "нет данных"}</span>
        <div class="r">${mName(month)} · доли трат</div>${sharesHtml(i)}`);
      if (!pinned) renderCard(d);
    }).on("mouseleave", hideTip)
      .on("click", (e, d) => { if (!row.has(d.id)) return; pinned = pinned === d ? null : d; sel(); renderCard(d); }));

    function renderCard(d) {
      const i = row.get(d.id), k = L.labels[i][month], v = S.values[i][month];
      const mx = Math.max(...v, ...ruMonth[month].map(x => x * sc)) / sc;
      $("#card").innerHTML = `<h3>${esc(d.properties.n)}</h3><div class="reg">${esc(d.properties.r)}</div>
        <span class="type-pill"><i style="background:${tColor(k)}"></i>${k >= 0 ? esc(T[k].name) : "нет данных"}</span>
        <div class="muted" style="font-size:12px">Доли трат, ${mName(month)}. Чёрная черта — среднее по всем МО.</div>
        <div class="bars">${CATS.map((c, j) => `<div class="lbl">${CAT_S[c]}</div>
          <div class="trk"><div class="val" style="width:${v[j] / sc / mx * 100}%"></div><div class="avg" style="left:${ruMonth[month][j] / mx * 100}%"></div></div>
          <div class="n">${P(v[j] / sc)}</div>`).join("")}</div>
        <div class="muted" style="font-size:12px">Тип по месяцам, 2023–2024</div>
        <div class="traj">${L.labels[i].map((t, j) => `<i title="${mName(j)}: ${t >= 0 ? esc(T[t].name) : "—"}" style="background:${tColor(t)}" class="${j === month ? "now" : ""}"></i>`).join("")}</div>
        <div class="traj-ax"><span>янв 2023</span><span>янв 2024</span><span>дек 2024</span></div>
        <p class="cap">В основном типе ${P(L.stability[i]).replace(",0", "")} месяцев · смен типа: ${L.switches[i]}${pinned === d ? " · закреплено" : ""}</p>`;
    }

    function renderLegend() {
      const el = $("#legend");
      if (layer === "stab") {
        el.innerHTML = `<div class="legend-h">Доля месяцев в основном типе</div>
          <div class="ramp" style="background:linear-gradient(90deg,${d3.range(0, 1.01, .25).map(t => stabColor(0.25 + t * 0.75)).join(",")})"></div>
          <div class="ramp-ax"><span>≤25 %</span><span>50 %</span><span>75 %</span><span>100 %</span></div>
          <p class="legend-note">Среднее по МО: ${P(d3.mean(L.stability))}. Все 24 месяца в одном типе: ${P(d3.mean(L.switches, s => s === 0))} МО.</p>`;
        return;
      }
      const cnt = count[month], mx = d3.max(cnt);
      el.innerHTML = `<div class="legend-h">Типы · МО в месяце</div>` + bySpend.map(k => { const t = T[k]; return `<button type="button" data-k="${k}" class="${isolate !== null && isolate !== k ? "off" : ""}" title="${esc(t.desc)}">
        <span class="sw" style="background:${t.color}"></span><span class="nm">${esc(short(t.name))}</span><span class="ct">${int(cnt[k])}</span>
        <span class="bar"><i style="width:${cnt[k] / mx * 100}%;background:${t.color}"></i></span></button>`; }).join("")
        + `<p class="legend-note"><span class="sw-inline"></span>серые — ${int(geo.features.length - M.n_mo)} МО без полных 24 месяцев. Нажмите на тип, чтобы оставить на карте только его.${namesStub ? " <b>Названия типов черновые.</b>" : ""}</p>`;
      el.querySelectorAll("button").forEach(b => b.onclick = () => { const k = +b.dataset.k; isolate = isolate === k ? null : k; paint(); renderLegend(); });
    }

    const slider = $("#month"), mLabel = $("#month-label");
    slider.max = NM - 1;
    const setMonth = m => { month = m; slider.value = m; mLabel.textContent = mName(m); paint(); renderLegend(); if (pinned) renderCard(pinned); };
    slider.oninput = () => setMonth(+slider.value);
    let timer = null;
    $("#play").onclick = () => {
      if (timer) { clearInterval(timer); timer = null; $("#play").textContent = "▶"; return; }
      $("#play").textContent = "❚❚";
      timer = setInterval(() => setMonth((month + 1) % NM), 800);
    };
    document.querySelectorAll("[data-layer]").forEach(b => b.onclick = () => {
      layer = b.dataset.layer; document.querySelectorAll("[data-layer]").forEach(x => x.classList.toggle("on", x === b)); paint(); renderLegend();
    });
    document.querySelectorAll("[data-geo]").forEach(b => b.onclick = () => b.dataset.geo !== form && setForm(b.dataset.geo));
    const byLabel = new Map(f2016.map(f => [`${f.properties.n} — ${f.properties.r}`, f]));
    $("#mo-list").innerHTML = [...byLabel.keys()].sort((a, b) => a.localeCompare(b, "ru")).map(s => `<option value="${esc(s)}">`).join("");
    $("#search").onchange = e => { const d = byLabel.get(e.target.value); if (!d) return; pinned = d; sel(); renderCard(d); zoomTo(d); };

    // Вводка: сколько площади карты занимает тип против его доли МО — отсюда и нужна картограмма
    const areaOf = new Map(geo.features.map(f => [f.id, path.area(f)])), totalArea = d3.sum(areaOf.values());
    const aShare = T.map((_, k) => d3.sum(f2016.filter((_, i) => modal[i] === k), f => areaOf.get(f.id)) / totalArea);
    const nShare = T.map((_, k) => d3.mean(modal, m => m === k));
    const big = d3.maxIndex(T, (_, k) => aShare[k] / nShare[k]);
    const msk = PLACES.find(p => p.label === "Москва");
    $("#map-lede").innerHTML = `На обычной карте тип «${esc(short(T[big].name))}» — <b>${P0(nShare[big])}</b> муниципалитетов, но <b>${P0(aShare[big])}</b> площади${msk ? `, а ${msk.idx.length} ${plural(msk.idx.length, "район", "района", "районов")} Москвы сливаются в точку` : ""}. Переключите на равные клетки: каждое МО — одна клетка, и видно, из чего на самом деле состоит страна.`;

    tiles.attr("transform", (_, i) => tf(i, "area"));
    places.attr("x", p => placeXY(p, "area")[0]).attr("y", p => placeXY(p, "area")[1] - 8);
    fontFix(); addEventListener("resize", fontFix);
    setMonth(0); setForm("area", false);
  })();

  // ================= 02. Переходы: аллювиальная диаграмма =================
  (function alluvial() {
    const F = A.transitions, Q = F.quarters.length;
    const W = 1180, H = 540, top = 40, left = 54, right = 250, nw = 8, gap = 12;
    const total = d3.sum(F.nodes[0]);
    const ky = (H - top - 12 - gap * (K - 1)) / total;
    const x = d3.scalePoint(d3.range(Q), [left, W - right - nw]);
    const s = d3.select("#alluvial").attr("viewBox", `0 0 ${W} ${H}`);
    const ord = bySpend, pos = new Map(ord.map((k, j) => [k, j]));  // сверху вниз — от высоких трат к низким
    const y0 = F.nodes.map(col => { let y = top; const out = []; ord.forEach(k => { out[k] = y; y += col[k] * ky + gap; }); return out; });
    const onlyMoves = $("#only-moves");
    const movers = F.links.filter(l => l.s !== l.t);
    const top3 = [...movers].sort((a, b) => b.n - a.n).slice(0, 3);
    const defs = s.append("defs");
    const draw = () => {
      s.selectAll("g").remove(); defs.selectAll("*").remove();
      const out = F.nodes.map(c => c.map(() => 0)), inn = F.nodes.map(c => c.map(() => 0));
      const links = [...F.links].sort((a, b) => a.q - b.q || pos.get(a.s) - pos.get(b.s) || pos.get(a.t) - pos.get(b.t));
      const band = links.map(l => { const h = l.n * ky, sy = y0[l.q][l.s] + out[l.q][l.s], ty = y0[l.q + 1][l.t]; out[l.q][l.s] += h; return { l, h, sy, ty }; });
      [...band].sort((a, b) => a.l.q - b.l.q || pos.get(a.l.t) - pos.get(b.l.t) || pos.get(a.l.s) - pos.get(b.l.s)).forEach(b => { b.ty += inn[b.l.q + 1][b.l.t]; inn[b.l.q + 1][b.l.t] += b.h; });
      const xa = q => x(q) + nw, xb = q => x(q + 1);
      const shown = band.filter(b => !onlyMoves.checked || b.l.s !== b.l.t);
      shown.forEach((b, i) => { if (b.l.s === b.l.t) return; b.gid = "lg" + i;
        const lg = defs.append("linearGradient").attr("id", b.gid).attr("gradientUnits", "userSpaceOnUse").attr("x1", xa(b.l.q)).attr("x2", xb(b.l.q));
        lg.append("stop").attr("offset", "15%").attr("stop-color", T[b.l.s].color); lg.append("stop").attr("offset", "85%").attr("stop-color", T[b.l.t].color); });
      const baseOp = b => b.l.s === b.l.t ? 0.4 : 0.72;
      const lk = s.append("g").selectAll("path").data(shown).join("path").attr("class", "link")
        .attr("d", b => { const a = xa(b.l.q), c = xb(b.l.q), m = (a + c) / 2;
          return `M${a},${b.sy}C${m},${b.sy} ${m},${b.ty} ${c},${b.ty}L${c},${b.ty + b.h}C${m},${b.ty + b.h} ${m},${b.sy + b.h} ${a},${b.sy + b.h}Z`; })
        .attr("fill", b => b.gid ? `url(#${b.gid})` : "#d9d3c6").attr("fill-opacity", baseOp)
        .on("mousemove", (e, b) => { focus(l => l === b.l);
          const from = F.nodes[b.l.q][b.l.s];
          showTip(e, `<b>${esc(short(T[b.l.s].name))} → ${esc(short(T[b.l.t].name))}</b><div class="r">${F.quarters[b.l.q]} → ${F.quarters[b.l.q + 1]}</div>${int(b.l.n)} МО · ${P(b.l.n / from)} типа`); })
        .on("mouseleave", () => { focus(null); hideTip(); });
      const focus = test => lk.attr("fill-opacity", b => !test ? baseOp(b) : test(b.l) ? (b.l.s === b.l.t ? 0.8 : 0.95) : 0.06);
      const hiType = k => k === null ? focus(null) : focus(l => l.s === k || l.t === k);
      const nodes = F.nodes.flatMap((col, q) => col.map((n, k) => ({ q, k, n })));
      s.append("g").selectAll("rect").data(nodes).join("rect")
        .attr("x", d => x(d.q)).attr("y", d => y0[d.q][d.k]).attr("width", nw).attr("height", d => Math.max(1, d.n * ky))
        .attr("fill", d => T[d.k].color)
        .on("mousemove", (e, d) => { hiType(d.k); showTip(e, `<b>${esc(short(T[d.k].name))}</b><div class="r">${F.quarters[d.q]}</div>${int(d.n)} МО`); })
        .on("mouseleave", () => { hiType(null); hideTip(); });
      s.append("g").selectAll("text").data(F.quarters).join("text").attr("class", "q")
        .attr("x", (_, q) => x(q) + nw / 2).attr("y", 18).attr("text-anchor", "middle").text(d => d.replace(" К", " · К"));
      const wrap = (t, n) => t.split(" ").reduce((ls, w) => { const l = ls[ls.length - 1]; (l && (l + " " + w).length <= n) ? ls[ls.length - 1] = l + " " + w : ls.push(w); return ls; }, []);
      const cy = (q, k) => y0[q][k] + F.nodes[q][k] * ky / 2;
      s.append("g").selectAll("text").data(ord).join("text").attr("x", x(0) - 8).attr("y", k => cy(0, k)).attr("dy", "0.35em")
        .attr("text-anchor", "end").text(k => int(F.nodes[0][k]));
      s.append("g").selectAll("text").data(ord).join("text").attr("class", "tn").attr("x", x(Q - 1) + nw + 10).attr("y", k => cy(Q - 1, k))
        .each(function (k) {
          const ls = wrap(short(T[k].name), 32), el = d3.select(this);
          ls.forEach((l, i) => el.append("tspan").attr("x", x(Q - 1) + nw + 10).attr("dy", i ? "1.15em" : `${0.35 - (ls.length - 1) * 0.575}em`).text(l));
          el.append("tspan").attr("class", "c").attr("dx", 6).text(int(F.nodes[Q - 1][k]));
        })
        .on("mousemove", (e, k) => hiType(k)).on("mouseleave", () => hiType(null));
      // Нумерованные выноски трёх самых больших смен типа — пояснение под диаграммой
      const ann = s.append("g").attr("class", "ann");
      top3.forEach((l, j) => {
        const b = band.find(b => b.l === l); if (!b || !shown.includes(b)) return;
        const mx = (xa(l.q) + xb(l.q)) / 2, my = (b.sy + b.ty) / 2 + b.h / 2;
        ann.append("circle").attr("cx", mx).attr("cy", my).attr("r", 10).attr("fill", "#fffdf8");
        ann.append("text").attr("class", "big").attr("x", mx).attr("y", my).attr("dy", "0.36em").attr("text-anchor", "middle").style("stroke-width", 0).text(j + 1);
      });
    };
    onlyMoves.onchange = draw; draw();
    const all = d3.sum(F.links, l => l.n), moved = d3.sum(movers, l => l.n);
    $("#flow-stat").textContent = `За квартал тип меняют в среднем ${P(moved / all)} МО.`;
    // Пары типов, между которыми больше всего переходов (в обе стороны)
    const pairs = d3.rollups(movers, v => d3.sum(v, l => l.n), l => [Math.min(l.s, l.t), Math.max(l.s, l.t)].join("-")).sort((a, b) => b[1] - a[1]);
    const nm = k => `«${esc(short(T[k].name))}»`;
    const p3 = pairs.slice(0, 3).map(([key, n]) => { const [a, b] = key.split("-").map(Number); return `${nm(a)} ⇄ ${nm(b)} (${int(n)})`; });
    const hiSet = new Set(bySpend.filter(k => (T[k].spend_ratio ?? 1) >= 1));
    const cross = d3.sum(movers.filter(l => hiSet.has(l.s) !== hiSet.has(l.t)), l => l.n);
    const share3 = d3.sum(pairs.slice(0, 3), p => p[1]) / moved;
    $("#flow-lede").innerHTML = `Смена типа — почти всегда шаг к соседнему профилю. Три пары дают <b>${P0(share3)}</b> всех переходов за семь кварталов: ${p3.join("; ")}. Между типами с тратами выше и ниже средних по стране — лишь <b>${P0(cross / moved)}</b> смен.`;
    const q = F.quarters;
    $("#flows .scroll-x").insertAdjacentHTML("afterend", `<ol class="callouts">${top3.map(l => `<li><b>${int(l.n)} МО</b> ${q[l.q]} → ${q[l.q + 1]}: из ${nm(l.s)} в ${nm(l.t)}</li>`).join("")}</ol>`);
  })();

  // ================= 03. Профили: маленькие множества =================
  (function profiles() {
    const lx = r => Math.max(0, Math.min(100, (Math.log2(r) + 1) * 50));  // ×0,5…×2 на одной оси у всех
    const portrait = d => (d || "").split(/(?<=[.!?])\s+(?=[А-ЯЁA-Z«])/)[0];
    const miniW = 300, fMini = fitter(eqXY, () => LY.r_eq, miniW, 100, 2), mh = Math.round(miniW / fMini.aspect) + 4;
    const fM = fitter(eqXY, () => LY.r_eq, miniW, mh, 2), mr = HEXR * fM.k * 0.92;
    const panel = k => {
      const t = T[k];
      const rows = CATS.map((c, j) => { const r = t.shares[j] / M.ru_shares[j], a = lx(r);
        return `<span class="l">${CAT_S[c]}</span><span class="trk"><span class="stem" style="left:${Math.min(a, 50)}%;width:${Math.abs(a - 50)}%"></span><span class="dot" style="left:${a}%"></span></span><span class="v${Math.abs(Math.log2(r)) > 0.4 ? " hi" : ""}" title="доля ${P(t.shares[j])}">${xr(r)}</span>`; }).join("");
      const ex = t.examples.slice(0, 3).map(e => esc(e.n.replace(/^(городской округ( город)?|муниципальный (район|округ)|город) /, "").replace(/ муниципальный (район|округ)$/, ""))).join(", ");
      return `<article class="prof" style="--c:${t.color}"><h3>${esc(short(t.name))}</h3>
        <p class="por">${esc(portrait(t.desc))}</p>
        <canvas width="${miniW * 2}" height="${mh * 2}" data-k="${k}" role="img" aria-label="Где тип «${esc(short(t.name))}» на картограмме равных клеток"></canvas>
        <dl class="nums"><div><dd>${int(t.n_mo)}</dd><dt>МО, основной<br>тип</dt></div>
          <div><dd>${t.growth_total == null ? "—" : sgn(t.growth_total)}</dd><dt>рост трат<br>2024 к 2023</dt></div>
          <div><dd>${t.wage_2023 == null ? "—" : ru.format(".1f")(t.wage_2023 / 1000)}</dd><dt>зарплата,<br>тыс. ₽, 2023</dt></div></dl>
        <div class="xplot">${rows}</div>
        <div class="xplot-ax"><span></span><span><i>×0,5</i><i>×1</i><i>×2</i></span><span></span></div>
        <p class="more">Траты на жителя <b>${t.spend_ratio == null ? "—" : xr(t.spend_ratio)}</b> к среднему · в своём типе <b>${P0(t.stability)}</b> месяцев · население (медиана) <b>${t.population_2024 == null ? "—" : int(t.population_2024)}</b> · доступность рынков <b>${t.market_access_2024 == null ? "—" : int(Math.round(t.market_access_2024))}</b>. Примеры: ${ex || "—"}.</p></article>`;
    };
    const key = `<article class="prof key"><h3>Как читать панель</h3>
      <p><b>Мини-карта</b> — картограмма равных клеток: цветом — МО, у которых это основной тип за два года.</p>
      <p><b>Три числа</b> — сколько МО, как быстро росли траты (все МО: ${sgn(M.ru_growth_total)}) и зарплата по Росстату.</p>
      <p><b>Точки на оси</b> — доля категории в тратах типа к средней по всем МО. Правее черты — больше, чем в среднем; шкала логарифмическая, ×2 и ×0,5 равноудалены.</p></article>`;
    $("#profiles").innerHTML = bySpend.map(panel).join("") + key;
    const others = d3.range(N);
    document.querySelectorAll(".prof canvas").forEach(cv => {
      const k = +cv.dataset.k, c = cv.getContext("2d"); c.scale(2, 2);
      const dotAt = (i, col) => { const [x, y] = fM(eqXY[i]); c.moveTo(x + mr, y); c.arc(x, y, mr, 0, 6.2832); };
      c.beginPath(); c.fillStyle = "#e2ddd2"; others.forEach(i => modal[i] !== k && dotAt(i)); c.fill();
      c.beginPath(); c.fillStyle = T[k].color; others.forEach(i => modal[i] === k && dotAt(i)); c.fill();
    });
  })();

  // ================= 04. Методы: хитмап оценок =================
  // Места не считаются здесь: они из outputs/method_ranking.csv (пороговое агрегирование и Борда, с AVU и без).
  (function methods() {
    const I = A.methods.indices, PR = A.methods.periods, monthCb = $("#m-month"), noAvuCb = $("#m-noavu");
    const val = (r, ix) => ix.by.startsWith("z_") ? r.z[ix.key] : r[ix.key];  // по чему ставится оценка
    const fmt = (k, v) => v == null ? "—" : k === "CH" ? int(Math.round(v)) : f3(v);
    const fz = z => Math.abs(z) >= 100 ? int(Math.round(z)) : ru.format(".1f")(z);
    const draw = () => {
      const R = PR[monthCb.checked ? "month_median" : "all"], noAvu = noAvuCb.checked;
      const pl = r => noAvu ? r.place_noavu : r.place;
      const best = Object.fromEntries(I.map(ix => [ix.key, (ix.better === "max" ? d3.max : d3.min)(R, r => val(r, ix))]));
      const rows = [...R].sort((a, b) => pl(a) - pl(b) || a.place - b.place);
      $("#m-state").textContent = (monthCb.checked ? "Индексы — медиана по 24 помесячным разбиениям" : "Индексы — по сквозным признакам за два года") + (noAvu ? "; места без AVU." : ".");
      $("#methods-table").innerHTML = `<caption class="sr">Индексы качества пяти методов при K = 7, оценки 1–3 по каждому индексу и итоговые места</caption>
        <thead><tr class="grp"><th></th>${["признаки", "граф"].map(on => `<th colspan="${I.filter(ix => ix.on === on).length}">на ${on === "граф" ? "графе сходства" : "признаках"}</th>`).join("")}<th></th></tr>
        <tr><th>Метод, K = 7</th>${I.map(ix => `<th class="${noAvu && ix.key === "AVU" ? "off" : ""}">${esc(ix.name)}<small>${ix.better === "max" ? "↑ больше — лучше" : "↓ меньше — лучше"}</small></th>`).join("")}<th style="text-align:center">Место</th></tr></thead>
        <tbody>${rows.map(r => `<tr class="${pl(r) === 1 ? "win" : ""} ${r.final ? "fin" : ""}"><td><span class="nm">${esc(r.name)}</span>${r.final ? '<span class="tag">итог</span>' : ""}<span class="fam">${esc(r.family)}</span></td>
          ${I.map(ix => { const off = noAvu && ix.key === "AVU";
            return `<td class="g${r.grade[ix.key]}${off ? " off" : ""}${!off && val(r, ix) === best[ix.key] ? " best" : ""}" title="оценка ${r.grade[ix.key]}">${fmt(ix.key, r[ix.key])}${ix.by.startsWith("z_") ? `<small>z = ${fz(r.z[ix.key])}</small>` : ""}</td>`; }).join("")}
          <td class="place" data-key="${esc(r.key)}"><span>${pl(r)}</span></td></tr>`).join("")}</tbody>`;
      $("#methods-table").querySelectorAll("td.place").forEach(td => {
        const r = R.find(x => x.key === td.dataset.key);
        const g = I.filter(ix => !(noAvu && ix.key === "AVU")).map(ix => r.grade[ix.key]);
        const n = v => g.filter(x => x === v).length;
        const html = `<b>${esc(r.name)}</b><div class="r">${monthCb.checked ? "медиана по 24 месяцам" : "сквозные признаки"}${noAvu ? ", без AVU" : ""}</div>
          Оценки по индексам (1 — верхняя треть): «1» ×${n(1)}, «2» ×${n(2)}, «3» ×${n(3)}.<br>Борда: ${noAvu ? r.borda_noavu : r.borda}-е место${noAvu ? "" : ` (${ru.format(".0f")(r.borda_points)} очков)`}.`;
        td.onmousemove = e => showTip(e, html); td.onclick = e => showTip(e, html); td.onmouseleave = hideTip;
      });
    };
    monthCb.onchange = noAvuCb.onchange = draw; draw();
    const rep = M.links.report;
    $("#avu-link").outerHTML = rep && rep !== "#" ? `<a href="${esc(rep)}#54-свойства-avu">отчёт, § 5.4</a>` : "отчёт, § 5.4";
  })();
})();
