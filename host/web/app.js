// App de lectura: muestra el texto, informa al servidor la posición de cada palabra y
// aplica las variantes que decide el servidor (siempre en páginas por delante del lector).
"use strict";

const estado = {
  texto: null, zoom: 1.5, variantes: {}, ws: null, verOriginal: false,
  altoPagina: 0, origenY: 0, oracionesPorId: {},
};

const $ = (s) => document.querySelector(s);
const cm = (n) => n * 96 / 2.54;          // px CSS por cm

// ------------------------------------------------------------------ conexión
function conectar() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => { enviarMarcadores(); enviarMaqueta(true); };
  ws.onmessage = (e) => recibir(JSON.parse(e.data));
  ws.onclose = () => setTimeout(conectar, 1000);
  estado.ws = ws;
}
function enviar(m) { if (estado.ws && estado.ws.readyState === 1) estado.ws.send(JSON.stringify(m)); }

function recibir(m) {
  if (m.tipo === "variantes") {
    for (const [sid, vid] of Object.entries(m.decisiones)) {
      estado.variantes[sid] = vid;
      pintarOracion(sid);
    }
    marcarPaginas();
    enviarMaqueta(true);
  } else if (m.tipo === "estado") {
    const chip = $("#estado");
    chip.textContent = `${m.estado} · carga ${m.carga}`;
    chip.className = `chip ${m.estado}`;
  } else if (m.tipo === "mirada") {
    const d = $("#mirada");
    if (!d.hidden) { d.style.left = `${m.x}px`; d.style.top = `${m.y}px`; }
  } else if (m.tipo === "calibracion") {
    const e = m.error_grados;
    $("#calib-msg").textContent = `Error medio: ojo izq. ${e.L ?? "—"}°, ojo der. ${e.R ?? "—"}°`;
    setTimeout(() => { $("#calibracion").hidden = true; }, 2500);
  }
}

// ----------------------------------------------------------------- render
async function cargar() {
  const r = await fetch("/api/texto");
  const d = await r.json();
  estado.texto = d.texto;
  estado.zoom = d.zoom;
  document.documentElement.style.setProperty("--z", d.zoom);
  $("#titulo").textContent = d.texto.titulo || "";
  const doc = $("#documento");
  const hoja = document.createElement("article");
  hoja.className = "hoja";
  let orden = 0;
  for (const cap of d.texto.capitulos) {
    if (cap.titulo) { const h = document.createElement("h2"); h.textContent = cap.titulo; hoja.append(h); }
    for (const par of cap.parrafos) {
      const p = document.createElement("p");
      for (const o of par.oraciones) {
        const span = document.createElement("span");
        span.className = "oracion";
        span.dataset.sid = o.id;
        span.dataset.orden = orden++;
        estado.oracionesPorId[o.id] = o;
        estado.variantes[o.id] = "orig";
        p.append(span, " ");
      }
      hoja.append(p);
    }
    if (cap.preguntas && cap.preguntas.length) hoja.append(bloquePreguntas(cap));
  }
  doc.append(hoja);
  for (const sid of Object.keys(estado.oracionesPorId)) pintarOracion(sid);
  estado.altoPagina = cm(24.7) * estado.zoom;          // 29,7 cm - 2 x 2,5 cm de margen
  marcarPaginas();
  conectar();
}

function pintarOracion(sid) {
  const span = document.querySelector(`.oracion[data-sid="${sid}"]`);
  const o = estado.oracionesPorId[sid];
  const vid = estado.verOriginal ? "orig" : estado.variantes[sid];
  const v = o.variantes.find((x) => x.id === vid) || o.variantes[0];
  span.replaceChildren();
  v.texto.split(/\s+/).filter(Boolean).forEach((w, i) => {
    const s = document.createElement("span");
    s.className = "w";
    s.dataset.i = i;
    s.textContent = w;
    span.append(s, " ");
  });
  span.dataset.vid = v.id;
  span.classList.toggle("modificada", v.id !== "orig");
  span.title = v.id !== "orig" ? `Original: ${o.variantes[0].texto}` : "";
}

function bloquePreguntas(cap) {
  const div = document.createElement("div");
  div.className = "preguntas";
  cap.preguntas.forEach((q, qi) => {
    const fs = document.createElement("fieldset");
    const lg = document.createElement("legend");
    lg.textContent = q.pregunta;
    fs.append(lg);
    q.opciones.forEach((op, oi) => {
      const id = `q-${cap.id}-${qi}-${oi}`;
      const lab = document.createElement("label");
      const inp = Object.assign(document.createElement("input"), { type: "radio", name: `q-${cap.id}-${qi}`, id });
      inp.addEventListener("change", () => enviar({ tipo: "respuesta", capitulo: cap.id, pregunta: qi,
        eleccion: oi, correcta: oi === q.correcta }));
      lab.htmlFor = id;
      lab.append(inp, " ", op);
      fs.append(lab, document.createElement("br"));
    });
    div.append(fs);
  });
  return div;
}

// Líneas punteadas cada "página" (alto útil de una hoja A4 con márgenes normales)
function marcarPaginas() {
  const hoja = $(".hoja");
  hoja.querySelectorAll(".separador-pagina").forEach((e) => e.remove());
  estado.origenY = hoja.getBoundingClientRect().top + scrollY + cm(2.5) * estado.zoom;
  const alto = hoja.scrollHeight;
  for (let k = 1; k * estado.altoPagina < alto; k++) {
    const sep = document.createElement("div");
    sep.className = "separador-pagina";
    sep.style.position = "absolute";
    sep.style.left = "0"; sep.style.right = "0";
    sep.style.top = `${cm(2.5) * estado.zoom + k * estado.altoPagina}px`;
    sep.innerHTML = `<span>pág. ${k + 1}</span>`;
    hoja.style.position = "relative";
    hoja.append(sep);
  }
}
const paginaDe = (yDoc) => Math.max(0, Math.floor((yDoc - estado.origenY) / estado.altoPagina));

// ------------------------------------------------------- maqueta para el servidor
// Envía las cajas de las palabras cercanas a la vista (coordenadas del documento).
function enviarMaqueta(reemplazar = false) {
  const vh = innerHeight, palabras = [], paginas = {};
  document.querySelectorAll(".oracion").forEach((sp) => {
    const r = sp.getBoundingClientRect();
    paginas[sp.dataset.sid] = paginaDe(r.top + scrollY);
    if (r.bottom < -vh || r.top > 2 * vh) return;
    const orden = Number(sp.dataset.orden);
    sp.querySelectorAll(".w").forEach((w) => {
      const b = w.getBoundingClientRect();
      palabras.push({ wid: `${sp.dataset.sid}|${sp.dataset.vid}|${w.dataset.i}`, sid: sp.dataset.sid,
        texto: w.textContent, orden: orden * 1000 + Number(w.dataset.i), pagina: paginaDe(b.top + scrollY),
        x0: b.left, y0: b.top + scrollY, x1: b.right, y1: b.bottom + scrollY });
    });
  });
  enviar({ tipo: "maqueta", palabras, paginas_oracion: paginas, reemplazar });
}

function enviarMarcadores() {
  const esquinas = {};
  document.querySelectorAll(".marcador").forEach((m) => {
    const r = m.getBoundingClientRect(), p = 6;   // el borde blanco no es parte del marcador
    esquinas[m.dataset.id] = [[r.left + p, r.top + p], [r.right - p, r.top + p],
      [r.right - p, r.bottom - p], [r.left + p, r.bottom - p]];
  });
  enviar({ tipo: "marcadores", esquinas });
}

let pendiente = null;
addEventListener("scroll", () => {
  enviar({ tipo: "scroll", y: scrollY });
  clearTimeout(pendiente);
  pendiente = setTimeout(() => enviarMaqueta(true), 150);
}, { passive: true });
addEventListener("resize", () => { marcarPaginas(); enviarMarcadores(); enviarMaqueta(true); });

// ------------------------------------------------------------------ calibración
async function calibrar() {
  try { await document.documentElement.requestFullscreen(); } catch (_) { /* opcional */ }
  await new Promise((r) => setTimeout(r, 400));
  enviarMarcadores();
  const capa = $("#calibracion"), punto = $("#punto");
  capa.hidden = false;
  $("#calib-msg").textContent = "Mire el punto rojo sin mover la cabeza.";
  enviar({ tipo: "calib_inicio" });
  const xs = [0.1, 0.5, 0.9], ys = [0.15, 0.5, 0.85];
  let idx = 0;
  for (const fy of ys) for (const fx of xs) {
    const x = fx * innerWidth, y = fy * innerHeight;
    punto.style.left = `${x}px`; punto.style.top = `${y}px`;
    enviar({ tipo: "calib_punto", idx: idx++, x, y });
    await new Promise((r) => setTimeout(r, 1600));
  }
  $("#calib-msg").textContent = "Calculando…";
  enviar({ tipo: "calib_fin" });
}

$("#btn-calibrar").addEventListener("click", calibrar);
$("#btn-original").addEventListener("click", (e) => {
  estado.verOriginal = !estado.verOriginal;
  e.currentTarget.setAttribute("aria-pressed", String(estado.verOriginal));
  e.currentTarget.textContent = estado.verOriginal ? "Ver adaptado" : "Ver original";
  Object.keys(estado.oracionesPorId).forEach(pintarOracion);
  enviarMaqueta(true);
});
addEventListener("keydown", (e) => { if (e.key === "g") $("#mirada").hidden = !$("#mirada").hidden; });

cargar();
