// Generic sheet probe. No http-server needed: the page is opened from file:// and every
// cdnjs MathJax request is answered from the local npm copy in ./vendor/package/es5.
//
// usage: node probe.js <dir containing index.html> <outdir> [--shots] [--expect expect.json]
//
//   --shots          screenshot every `section .lab` at 1180 and 390 into <outdir>
//   --expect f.json  [{"id":"bH","value":30,"tol":0}, {"id":"bD","re":"3\\.16"}] checked AT REST
//                    (before any interaction) at 1180. value: first number in the text,
//                    commas stripped; re: regex on the raw text.
//
// Exit code 1 if any check fails. Read the screenshots anyway: structure passing proves
// nothing is broken, only a screenshot shows whether anything is legible.
const path = require('path'), fs = require('fs');
const { chromium } = require('/opt/node22/lib/node_modules/playwright');
// MathJax 3.2.2 es5 tree: set MATHJAX_ES5, or unpack `npm pack mathjax@3.2.2` next to this file.
const VENDOR = process.env.MATHJAX_ES5 || path.join(__dirname, 'vendor/package/es5');
const dir = path.resolve(process.argv[2]);
const out = path.resolve(process.argv[3] || '/tmp/probe-out');
const shots = process.argv.includes('--shots');
const ei = process.argv.indexOf('--expect');
const expect = ei > 0 ? JSON.parse(fs.readFileSync(process.argv[ei + 1], 'utf8')) : [];
fs.mkdirSync(out, { recursive: true });

// static checks on the source
const src = fs.readFileSync(path.join(dir, 'index.html'), 'utf8');
const fails = [];
const so = (src.match(/<style>/g) || []).length, sc = (src.match(/<\/style>/g) || []).length;
if (so !== sc) fails.push(`static: <style> ${so} != </style> ${sc}`);
const mb = (src.match(/<div class="mathbox[^"]*">/g) || []).length;
const mbOk = (src.match(/<div class="mathbox[^"]*">\s*(\$\$|\\\[)/g) || []).length;
if (mb !== mbOk) fails.push(`static: ${mb - mbOk} mathbox(es) not opening with $$ (unquoted heredoc?)`);
if (/\b\d{3,7}\\begin|class="mathbox[^"]*">\d{3,}/.test(src)) fails.push('static: PID-looking digits in a mathbox');
if (src.includes('@SRC@')) fails.push('static: unsubstituted @SRC@ placeholder');

(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' });
  for (const w of [1180, 900, 700, 480, 390]) {
    const ctx = await browser.newContext({ viewport: { width: w, height: 900 } });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push('pageerror: ' + String(e)));
    page.on('console', m => { if (m.type() === 'error' && !/ERR_|Failed to load resource/.test(m.text())) errs.push('console: ' + m.text()); });
    await page.route('**/*', route => {
      const u = route.request().url();
      const m = u.match(/mathjax\/[^/]+\/es5\/([^?#]*)/);
      if (m) {
        const f = path.join(VENDOR, m[1]);
        if (fs.existsSync(f)) return route.fulfill({ path: f, contentType: 'application/javascript' });
      }
      if (u.startsWith('file:') || u.startsWith('data:')) return route.continue();
      return route.abort();
    });
    await page.goto('file://' + path.join(dir, 'index.html'));
    await page.waitForFunction(() => window.MathJax && MathJax.startup && MathJax.startup.promise, null, { timeout: 15000 }).catch(() => {});
    await page.evaluate(() => MathJax.startup.promise).catch(() => {});
    await page.waitForTimeout(1200);

    if (w === 1180 && expect.length) {
      const got = await page.evaluate(ex => ex.map(e => {
        const el = document.getElementById(e.id); return el ? el.textContent : null;
      }), expect);
      expect.forEach((e, i) => {
        const t = got[i];
        let ok = t !== null;
        if (ok && e.re !== undefined) ok = new RegExp(e.re).test(t);
        if (ok && e.value !== undefined) {
          const m = t.replace(/,/g, '').replace(/−/g, '-').match(/-?\d+(\.\d+)?(e-?\d+)?/i);
          ok = !!m && Math.abs(parseFloat(m[0]) - e.value) <= (e.tol || 0);
        }
        console.log(`  expect #${e.id} = ${JSON.stringify(t)} ${ok ? 'ok' : '*** MISMATCH ***'} (want ${e.value ?? e.re}${e.tol ? ' ±' + e.tol : ''})`);
        if (!ok) fails.push(`expect #${e.id}: got ${JSON.stringify(t)}`);
      });
    }

    // drive every control: predict buttons, preset buttons, sliders end to end, selects
    await page.evaluate(async () => {
      const fire = (el, t) => el.dispatchEvent(new Event(t, { bubbles: true }));
      for (const b of document.querySelectorAll('main button:not(#themebtn)')) { try { b.click(); } catch (e) {} }
      for (const s of document.querySelectorAll('main input[type=range]')) {
        const v0 = s.value;
        for (const v of [s.min, s.max, (Number(s.min) + Number(s.max)) / 2, v0]) { s.value = v; fire(s, 'input'); fire(s, 'change'); }
      }
      for (const s of document.querySelectorAll('main select')) {
        for (const o of s.options) { s.value = o.value; fire(s, 'change'); }
      }
      for (const c of document.querySelectorAll('main input[type=checkbox]')) { c.click(); c.click(); }
    });
    await page.waitForTimeout(600);

    const r = await page.evaluate(() => {
      const de = document.documentElement;
      const res = { mjx: document.querySelectorAll('mjx-container').length, hscroll: de.scrollWidth - de.clientWidth };
      res.eqOver = [...document.querySelectorAll('.mathbox')].filter(b => {
        const c = b.querySelector('mjx-container'); return c && c.getBoundingClientRect().width > b.clientWidth + 1;
      }).map(b => (b.closest('section') || {}).id);
      res.clipped = [...document.querySelectorAll('.ro-k,.ro-v,.lg,.lab-t,.lab-tag,.tb-v,.btn,.pbtn')]
        .filter(e => e.scrollWidth > e.clientWidth + 1).map(e => e.className + ':' + e.textContent.trim().slice(0, 40));
      res.broken = [...document.images].filter(i => i.complete && i.naturalWidth === 0).map(i => i.getAttribute('src'));
      res.placeholder = [...document.querySelectorAll('main [id]')].filter(e => e.children.length === 0 && /^\s*(—|—|-|&mdash;)\s*$/.test(e.textContent)).map(e => e.id);
      res.wide = [...document.querySelectorAll('main *')].filter(e => e.getBoundingClientRect().right > de.clientWidth + 1 && !e.closest('.mathbox,.tw,mjx-container'))
        .slice(0, 5).map(e => e.tagName + '.' + e.className + '#' + e.id);
      return res;
    });
    const bad = [];
    if (r.mjx < 20) bad.push(`only ${r.mjx} mjx-container (MathJax did not run?)`);
    if (r.hscroll > 0) bad.push(`page scrolls sideways by ${r.hscroll}px; widest: ${r.wide.join(' ')}`);
    if (r.eqOver.length) bad.push(`display eq wider than .mathbox in: ${r.eqOver.join(',')}`);
    if (r.clipped.length) bad.push(`clipped: ${r.clipped.join(' | ')}`);
    if (r.broken.length) bad.push(`broken img: ${r.broken.join(',')}`);
    if (r.placeholder.length) bad.push(`readouts never written: ${r.placeholder.join(',')}`);
    if (errs.length) bad.push(`errors: ${[...new Set(errs)].join(' || ')}`);
    console.log(`w=${w}  mjx=${r.mjx}  ${bad.length ? '\n    ' + bad.join('\n    ') : 'clean'}`);
    bad.forEach(b => fails.push(`w=${w}: ${b}`));

    if (shots && (w === 1180 || w === 390)) {
      // reload so the shots show each lab AT REST, as a reader first meets it; hide the sticky
      // titleblock so it does not sit on top of the lab being photographed
      await page.reload();
      await page.evaluate(() => MathJax.startup.promise).catch(() => {});
      await page.waitForTimeout(1000);
      await page.addStyleTag({ content: '.titleblock{visibility:hidden !important}' });
      const labs = await page.$$('section .lab');
      for (let i = 0; i < labs.length; i++) {
        const sec = await labs[i].evaluate(e => (e.closest('section') || {}).id || 'x');
        await labs[i].scrollIntoViewIfNeeded();
        await labs[i].screenshot({ path: path.join(out, `w${w}-${String(i + 1).padStart(2, '0')}-${sec}.png`) });
      }
      await page.screenshot({ path: path.join(out, `w${w}-top.png`) });
      console.log(`  ${labs.length} lab screenshots -> ${out}`);
    }
    await ctx.close();
  }
  await browser.close();
  console.log(fails.length ? `\nPROBE FAILED (${fails.length}):\n  ` + fails.join('\n  ') : '\nPROBE CLEAN');
  process.exit(fails.length ? 1 : 0);
})();
