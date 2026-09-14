# Doc build tools

Regenerate `HLD.docx` and `HLD.pdf` from `HLD.md`.

## One-time setup
```
cd tools
npm init -y
npm i docx markdown-it puppeteer
```

## Build
```
cd tools
node convert.js ../HLD.md ..          # writes ../HLD.docx and ../HLD.html
node html2pdf.js ../HLD.html ../HLD.pdf
rm ../HLD.html
```

## Regenerate diagram images (after editing diagrams/*.mmd)
```
npm i @mermaid-js/mermaid-cli
npx mmdc -i ../diagrams/01-system-context.mmd -o ../diagrams/01-system-context.png -b white -s 2 -p puppeteer-config.json
npx mmdc -i ../diagrams/01-system-context.mmd -o ../diagrams/01-system-context.svg -b white -p puppeteer-config.json
```

`convert.js` strips the collapsible `<details>` Mermaid-source blocks and embeds `diagrams/*.png` inline.
