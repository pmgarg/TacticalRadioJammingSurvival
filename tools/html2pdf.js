const puppeteer = require('puppeteer');
const fs = require('fs');
(async () => {
  const [,, IN, OUT] = process.argv;
  const html = fs.readFileSync(IN, 'utf8');
  const browser = await puppeteer.launch({ args: ['--no-sandbox', '--disable-setuid-sandbox'] });
  const page = await browser.newPage();
  await page.setContent(html, { waitUntil: 'networkidle0' });
  await page.pdf({
    path: OUT, format: 'A4', printBackground: true,
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="width:100%;font-size:8px;color:#888;padding:0 16mm;text-align:right;">HLD — Jamming Survival: On-Device Mesh Agent · v0.6 · page <span class="pageNumber"></span>/<span class="totalPages"></span></div>',
  });
  await browser.close();
  console.log('wrote', OUT, fs.statSync(OUT).size, 'bytes');
})();
