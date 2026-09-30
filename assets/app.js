// Экономический атлас: одна страница, данные из data/atlas.js (собирает scripts/build_site.py).
(() => {
  const A = window.ATLAS;
  if (!A || !A.meta) { document.body.insertAdjacentHTML("afterbegin", "<p style='padding:24px'>Нет data/atlas.js — запустите scripts/build_site.py.</p>"); return; }
  const M = A.meta, T = M.types, CATS = M.categories, NM = M.months.length;
  const ru = d3.formatLocale({ decimal: ",", thousands: " ", grouping: [3], currency: ["", " ₽"] });
  const pct = ru.format(".1%"), int = ru.format(",d"), f2 = ru.format(".2f"), f3 = ru.format(".3f");
  const sgn = x => (x > 0 ? "+" : x < 0 ? "−" : "") + pct(Math.abs(x)).replace("%", " %");
  const P = x => pct(x).replace("%", " %");
  const MONTHS = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"];
  const mName = i => { const [y, m] = M.months[i].split("-"); return `${MONTHS[+m - 1]} ${y}`; };
  const CAT_S = { "Продовольствие": "Продукты", "Маркетплейсы": "Маркетплейсы", "Общественное питание": "Кафе и рестораны", "Здоровье": "Здоровье", "Транспорт": "Транспорт" };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const $ = s => document.querySelector(s);
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

  const L = A.labels, S = A.shares, sc = S.scale;
  const row = new Map(L.ids.map((id, i) => [id, i]));
  // Названия из types.csv длинные («Москва и ближнее Подмосковье: самые высокие траты…»): коротко — до двоеточия.
  const short = n => n.split(":")[0].trim(), rest = n => n.includes(":") ? n.slice(n.indexOf(":") + 1).trim() : "";
  const tColor = k => (k >= 0 && T[k] ? T[k].color : "#d9d6cf");
  const ruMonth = d3.range(NM).map(t => CATS.map((_, c) => d3.mean(S.values, v => v[t][c]) / sc));

  // ---- Шапка, пометка заглушек, факты
  // Всё настоящее, кроме названий типов до outputs/types.csv: помечаем только их, там, где они крупно.
  const namesStub = (M.stub || []).includes("названия типов");
  const STUB = namesStub ? `<p class="stub-note"><b>Черновые названия.</b> «Тип A…${String.fromCharCode(64 + T.length)}» — временные метки; содержательные названия появятся после разбора профилей. Состав типов и все числа — настоящие.</p>` : "";
  $("#types .sec-head").insertAdjacentHTML("afterend", STUB);
  $("#h-n").textContent = int(M.n_mo);
  $("#headline").textContent = M.headline;
  $("#built").textContent = M.built;
  $("#facts").innerHTML = [["муниципалитетов", int(M.n_mo)], ["регионов", M.n_regions], ["месяцев", NM], ["типов экономики", T.length], ["методов × индексов", `${A.methods.periods.all.length}×${A.methods.indices.length}`]]
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  const lk = M.links, link = (u, t) => u && u !== "#" ? `<a href="${esc(u)}">${t}</a>` : `<span title="Появится к сдаче">${t} — скоро</span>`;
  $("#links").innerHTML = link(lk.repo, "Репозиторий с кодом") + link(lk.report, "Отчёт") + link(lk.pdf, "PDF-версия");

  // ---- Карта
  const svg = d3.select("#map-svg"), W = 1000, H = 520;
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const g = svg.append("g");
  const geo = A.geo;
  const proj = d3.geoIdentity().reflectY(true).fitExtent([[10, 10], [W - 10, H - 10]], geo);
  const path = d3.geoPath(proj);
  let month = 0, layer = "type", isolate = null, pinned = null;
  const stabColor = d3.scaleSequential([0.25, 1], d3.interpolateRgbBasis(["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])).clamp(true);
  const paths = g.selectAll("path").data(geo.features).join("path").attr("d", path);
  const fill = d => {
    const i = row.get(d.id); if (i === undefined) return "#e4e1da";
    if (layer === "stab") return stabColor(L.stability[i]);
    const k = L.labels[i][month];
    return isolate === null || isolate === k ? tColor(k) : "#e7e4dd";
  };
  const paint = () => paths.attr("fill", fill);

  const zoom = d3.zoom().scaleExtent([1, 40]).translateExtent([[0, 0], [W, H]])
    .filter(e => e.type === "wheel" ? (e.ctrlKey || e.metaKey) : e.type.startsWith("touch") ? e.touches.length > 1 : !e.button)
    .on("zoom", e => g.attr("transform", e.transform));
  svg.call(zoom).style("touch-action", "pan-y");
  document.querySelectorAll(".zoom button").forEach(b => b.onclick = () => {
    const z = b.dataset.z;
    if (z === "reset") svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);
    else svg.transition().duration(300).call(zoom.scaleBy, z === "in" ? 2 : 0.5);
  });
  const zoomTo = d => {
    const [[x0, y0], [x1, y1]] = path.bounds(d);
    const k = Math.min(20, 0.35 / Math.max((x1 - x0) / W, (y1 - y0) / H));
    svg.transition().duration(700).call(zoom.transform, d3.zoomIdentity.translate(W / 2, H / 2).scale(k).translate(-(x0 + x1) / 2, -(y0 + y1) / 2));
  };

  const sharesHtml = i => {
    const v = S.values[i][month];
    return "<table>" + CATS.map((c, j) => `<tr><td>${CAT_S[c]}</td><td>${P(v[j] / sc)}</td></tr>`).join("") + "</table>";
  };
  paths.on("mousemove", (e, d) => {
    const i = row.get(d.id);
    if (i === undefined) return showTip(e, `<b>${esc(d.properties.n)}</b><div class="r">${esc(d.properties.r)}</div>Нет полных 24 месяцев данных — в типологию не входит.`);
    const k = L.labels[i][month];
    showTip(e, `<b>${esc(d.properties.n)}</b><div class="r">${esc(d.properties.r)}</div>
      <span class="type-pill"><i style="background:${tColor(k)}"></i>${k >= 0 ? esc(T[k].name) : "нет данных"}</span>
      <div class="r">${mName(month)} · доли трат</div>${sharesHtml(i)}`);
    if (!pinned) renderCard(d);
  }).on("mouseleave", hideTip)
    .on("click", (e, d) => { if (!row.has(d.id)) return; pinned = pinned === d ? null : d; paths.classed("sel", p => p === pinned); if (pinned) d3.select(e.currentTarget).raise(); renderCard(d); });

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
      el.innerHTML = `<div class="nm" style="margin:0 8px;font-weight:600">Доля месяцев в основном типе</div>
        <div class="ramp" style="background:linear-gradient(90deg,${d3.range(0, 1.01, .25).map(t => stabColor(0.25 + t * 0.75)).join(",")})"></div>
        <div class="ramp-ax"><span>≤25 %</span><span>50 %</span><span>75 %</span><span>100 %</span></div>
        <p class="legend-note">Среднее по МО: ${P(d3.mean(L.stability))}. Все 24 месяца в одном типе: ${P(d3.mean(L.switches, s => s === 0))} МО.</p>`;
      return;
    }
    const cnt = T.map((_, k) => L.labels.reduce((a, r) => a + (r[month] === k), 0));
    el.innerHTML = T.map((t, k) => `<button type="button" data-k="${k}" class="${isolate !== null && isolate !== k ? "off" : ""}">
      <span class="sw" style="background:${t.color}"></span><span class="nm" title="${esc(t.desc)}">${esc(short(t.name))}<small>${esc(rest(t.name))}</small></span><span class="ct">${int(cnt[k])}</span></button>`).join("")
      + `<p class="legend-note"><span class="sw-inline"></span>серые — ${int(geo.features.length - M.n_mo)} МО без полных 24 месяцев. Числа — МО в месяце «${mName(month)}». Нажмите на тип, чтобы оставить на карте только его.${namesStub ? " <b>Названия типов черновые.</b>" : ""}</p>`;
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
    timer = setInterval(() => setMonth((month + 1) % NM), 700);
  };
  document.querySelectorAll(".seg button").forEach(b => b.onclick = () => {
    layer = b.dataset.layer; document.querySelectorAll(".seg button").forEach(x => x.classList.toggle("on", x === b)); paint(); renderLegend();
  });
  const byLabel = new Map(geo.features.filter(f => row.has(f.id)).map(f => [`${f.properties.n} — ${f.properties.r}`, f]));
  $("#mo-list").innerHTML = [...byLabel.keys()].sort((a, b) => a.localeCompare(b, "ru")).map(s => `<option value="${esc(s)}">`).join("");
  $("#search").onchange = e => {
    const d = byLabel.get(e.target.value); if (!d) return;
    pinned = d; paths.classed("sel", p => p === d); paths.filter(p => p === d).raise(); renderCard(d); zoomTo(d);
  };
  setMonth(0);

  // ---- Аллювиальная диаграмма
  (function alluvial() {
    const F = A.transitions, Q = F.quarters.length, K = T.length;
    const W = 1100, H = 460, top = 30, left = 70, right = 230, nw = 14, gap = 8;
    const total = d3.sum(F.nodes[0]);
    const ky = (H - top - 10 - gap * (K - 1)) / total;
    const x = d3.scalePoint(d3.range(Q), [left, W - right - nw]);
    const s = d3.select("#alluvial").attr("viewBox", `0 0 ${W} ${H}`);
    const y0 = F.nodes.map(col => { let y = top; return col.map(n => { const v = y; y += n * ky + gap; return v; }); });
    const onlyMoves = $("#only-moves");
    const draw = () => {
      s.selectAll("*").remove();
      const out = F.nodes.map(c => c.map(() => 0)), inn = F.nodes.map(c => c.map(() => 0));
      const links = [...F.links].sort((a, b) => a.q - b.q || a.s - b.s || a.t - b.t);
      const band = links.map(l => {
        const h = l.n * ky, sy = y0[l.q][l.s] + out[l.q][l.s], ty = y0[l.q + 1][l.t];
        out[l.q][l.s] += h; return { l, h, sy, ty };
      });
      // входящие смещения — по порядку типа-источника, чтобы ленты не перекрещивались внутри столбика
      [...band].sort((a, b) => a.l.q - b.l.q || a.l.t - b.l.t || a.l.s - b.l.s).forEach(b => { b.ty += inn[b.l.q + 1][b.l.t]; inn[b.l.q + 1][b.l.t] += b.h; });
      const xa = q => x(q) + nw, xb = q => x(q + 1);
      s.append("g").selectAll("path").data(band.filter(b => !onlyMoves.checked || b.l.s !== b.l.t)).join("path")
        .attr("class", "link")
        .attr("d", b => { const a = xa(b.l.q), c = xb(b.l.q), m = (a + c) / 2;
          return `M${a},${b.sy}C${m},${b.sy} ${m},${b.ty} ${c},${b.ty}L${c},${b.ty + b.h}C${m},${b.ty + b.h} ${m},${b.sy + b.h} ${a},${b.sy + b.h}Z`; })
        .attr("fill", b => T[b.l.s].color).attr("fill-opacity", b => b.l.s === b.l.t ? 0.16 : 0.55)
        .on("mousemove", (e, b) => { d3.select(e.currentTarget).attr("fill-opacity", 0.85);
          const from = F.nodes[b.l.q][b.l.s];
          showTip(e, `<b>${esc(T[b.l.s].name)} → ${esc(T[b.l.t].name)}</b><div class="r">${F.quarters[b.l.q]} → ${F.quarters[b.l.q + 1]}</div>${int(b.l.n)} МО · ${P(b.l.n / from)} типа`); })
        .on("mouseleave", (e, b) => { d3.select(e.currentTarget).attr("fill-opacity", b.l.s === b.l.t ? 0.16 : 0.55); hideTip(); });
      const nodes = F.nodes.flatMap((col, q) => col.map((n, k) => ({ q, k, n })));
      s.append("g").selectAll("rect").data(nodes).join("rect")
        .attr("x", d => x(d.q)).attr("y", d => y0[d.q][d.k]).attr("width", nw).attr("height", d => Math.max(1, d.n * ky)).attr("rx", 2)
        .attr("fill", d => T[d.k].color)
        .on("mousemove", (e, d) => showTip(e, `<b>${esc(T[d.k].name)}</b><div class="r">${F.quarters[d.q]}</div>${int(d.n)} МО`)).on("mouseleave", hideTip);
      s.append("g").selectAll("text").data(F.quarters).join("text").attr("class", "q")
        .attr("x", (_, q) => x(q) + nw / 2).attr("y", 16).attr("text-anchor", "middle").text(d => d);
      // слева только числа, справа — короткое название в две строки и число
      const wrap = (t, n) => t.split(" ").reduce((ls, w) => { const l = ls[ls.length - 1]; (l && (l + " " + w).length <= n) ? ls[ls.length - 1] = l + " " + w : ls.push(w); return ls; }, []);
      const cy = (q, k) => y0[q][k] + F.nodes[q][k] * ky / 2;
      s.append("g").selectAll("text").data(T).join("text").attr("x", x(0) - 8).attr("y", (_, k) => cy(0, k)).attr("dy", "0.35em")
        .attr("text-anchor", "end").text((_, k) => int(F.nodes[0][k]));
      s.append("g").selectAll("text").data(T).join("text").attr("x", x(Q - 1) + nw + 8).attr("y", (_, k) => cy(Q - 1, k))
        .each(function (t, k) {
          const ls = wrap(short(t.name), 30), el = d3.select(this);
          ls.forEach((l, i) => el.append("tspan").attr("x", x(Q - 1) + nw + 8).attr("dy", i ? "1.2em" : `${0.35 - (ls.length - 1) * 0.6}em`)
            .text(i === ls.length - 1 ? `${l} · ${int(F.nodes[Q - 1][k])}` : l));
        });
    };
    onlyMoves.onchange = draw; draw();
    const moved = d3.sum(F.links.filter(l => l.s !== l.t), l => l.n), all = d3.sum(F.links, l => l.n);
    $("#flow-stat").textContent = `За квартал тип меняют в среднем ${P(moved / all)} МО.`;
  })();

  // ---- Профили типов
  (function profiles() {
    const lx = d3.scaleLog([0.5, 2], [0, 100]).clamp(true);
    $("#profiles").innerHTML = T.map(t => {
      const rows = CATS.map((c, j) => {
        const r = t.shares[j] / M.ru_shares[j], a = lx(Math.min(r, 1)), b = lx(Math.max(r, 1));
        return `<span>${CAT_S[c]}</span><span class="v">${P(t.shares[j])}</span>
          <span class="trk"><span class="bar" style="left:${a}%;width:${Math.max(0.8, b - a)}%"></span></span><span class="x">×${f2(r)}</span>`;
      }).join("");
      const ex = t.examples.map(e => `<b>${esc(e.n)}</b> (${esc(e.r)})`).join("; ");
      return `<article class="prof" style="--c:${t.color}"><h3>${esc(short(t.name))}</h3>${rest(t.name) ? `<p class="rest">${esc(rest(t.name))}</p>` : ""}<p class="desc">${esc(t.desc)}</p>
        <div class="meta"><span><b>${int(t.n_mo)}</b> МО с этим основным типом</span><span>в основном типе <b>${P(t.stability)}</b> месяцев</span></div>
        <div class="ratio">${rows}</div>
        <div class="ratio-ax"><span></span><span></span><span><i>×0,5</i><i>×1</i><i>×2</i></span><span></span></div>
        <dl><dt>Рост трат, 2024 к 2023</dt><dd>${t.growth_total == null ? "—" : sgn(t.growth_total)} <span class="muted">все МО ${sgn(M.ru_growth_total)}</span></dd>
          <dt>Средняя зарплата, 2023</dt><dd>${t.wage_2023 == null ? "—" : int(Math.round(t.wage_2023 / 100) * 100) + " ₽"}</dd>
          <dt>Население, 2024</dt><dd>${t.population_2024 == null ? "—" : int(t.population_2024)}</dd>
          <dt>Доступность рынка, 2024</dt><dd>${t.market_access_2024 == null ? "—" : t.market_access_2024 >= 100 ? int(Math.round(t.market_access_2024)) : f3(t.market_access_2024)}</dd></dl>
        <p class="ex">Примеры: ${ex || "—"}</p></article>`;
    }).join("");
  })();

  // ---- Таблица методов
  // Места не считаются здесь: они из outputs/method_ranking.csv (пороговое агрегирование и Борда, с AVU и без).
  (function methods() {
    const I = A.methods.indices, P = A.methods.periods, monthCb = $("#m-month"), noAvuCb = $("#m-noavu");
    const val = (r, ix) => ix.by.startsWith("z_") ? r.z[ix.key] : r[ix.key];  // по чему ставится оценка
    const fmt = (k, v) => v == null ? "—" : k === "CH" ? int(Math.round(v)) : f3(v);
    const fz = z => Math.abs(z) >= 100 ? int(Math.round(z)) : ru.format(".1f")(z);
    const draw = () => {
      const R = P[monthCb.checked ? "month_median" : "all"], noAvu = noAvuCb.checked;
      const pl = r => noAvu ? r.place_noavu : r.place;
      const best = Object.fromEntries(I.map(ix => [ix.key, (ix.better === "max" ? d3.max : d3.min)(R, r => val(r, ix))]));
      const rows = [...R].sort((a, b) => pl(a) - pl(b) || a.place - b.place);
      $("#m-state").textContent = (monthCb.checked ? "Индексы — медиана по 24 помесячным разбиениям" : "Индексы — по сквозным признакам за два года")
        + (noAvu ? "; места без AVU." : ".");
      $("#methods-table").innerHTML = `<caption class="sr">Индексы качества пяти методов при K = 7 и итоговые места</caption>
        <thead><tr class="grp"><th></th>${["признаки", "граф"].map(on => `<th colspan="${I.filter(ix => ix.on === on).length}">на ${on === "граф" ? "графе сходства" : "признаках"}</th>`).join("")}<th></th></tr>
        <tr><th>Метод, K = 7</th>${I.map(ix => `<th class="${noAvu && ix.key === "AVU" ? "off" : ""}">${esc(ix.name)}<small>${ix.better === "max" ? "↑ больше — лучше" : "↓ меньше — лучше"}</small></th>`).join("")}<th>Место</th></tr></thead>
        <tbody>${rows.map(r => `<tr class="${pl(r) === 1 ? "win" : ""}"><td>${esc(r.name)}${r.final ? '<span class="tag">итог</span>' : ""}<span class="fam">${esc(r.family)}</span></td>
          ${I.map(ix => { const off = noAvu && ix.key === "AVU";
            return `<td class="${off ? "off" : val(r, ix) === best[ix.key] ? "best" : ""}">${fmt(ix.key, r[ix.key])}${ix.by.startsWith("z_") ? `<small>z = ${fz(r.z[ix.key])}</small>` : ""}</td>`; }).join("")}
          <td class="place" data-key="${esc(r.key)}">${pl(r)}</td></tr>`).join("")}</tbody>`;
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
