/**
 * Podgląd PDF w stylu dokumentu (PDF.js) — bez natywnego paska przeglądarki.
 */
(function (global) {
  const PDFJS_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174';

  let pdfjsReady = null;

  function ensurePdfJs() {
    if (global.pdfjsLib) return Promise.resolve(global.pdfjsLib);
    if (pdfjsReady) return pdfjsReady;

    pdfjsReady = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = `${PDFJS_CDN}/pdf.min.js`;
      script.onload = () => {
        global.pdfjsLib.GlobalWorkerOptions.workerSrc = `${PDFJS_CDN}/pdf.worker.min.js`;
        resolve(global.pdfjsLib);
      };
      script.onerror = () => reject(new Error('Nie udało się załadować PDF.js'));
      document.head.appendChild(script);
    });
    return pdfjsReady;
  }

  class PdfViewer {
    constructor(root) {
      this.root = root;
      this.pdfDoc = null;
      this.manualScale = 1;
      this.fitMode = true;
      this.renderGen = 0;
      this._resizeObs = null;

      this.root.innerHTML = `
        <div class="pdf-v-toolbar">
          <div class="pdf-v-toolbar-left">
            <button type="button" class="pdf-v-btn" data-action="prev" title="Poprzednia strona" disabled>
              <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M15 19l-7-7 7-7"/></svg>
            </button>
            <span class="pdf-v-page-info" data-page-info>—</span>
            <button type="button" class="pdf-v-btn" data-action="next" title="Następna strona" disabled>
              <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 5l7 7-7 7"/></svg>
            </button>
          </div>
          <div class="pdf-v-toolbar-right">
            <button type="button" class="pdf-v-btn" data-action="zoom-out" title="Pomniejsz">−</button>
            <button type="button" class="pdf-v-btn pdf-v-btn-text" data-action="fit" title="Dopasuj szerokość">Dopasuj</button>
            <button type="button" class="pdf-v-btn" data-action="zoom-in" title="Powiększ">+</button>
          </div>
        </div>
        <div class="pdf-v-stage" data-stage>
          <div class="pdf-v-loading" data-loading>
            <div class="pdf-v-spinner"></div>
            <p>Ładowanie dokumentu…</p>
          </div>
          <div class="pdf-v-scroll" data-scroll hidden>
            <div class="pdf-v-pages" data-pages></div>
          </div>
        </div>`;

      this.stage = root.querySelector('[data-stage]');
      this.loadingEl = root.querySelector('[data-loading]');
      this.scrollEl = root.querySelector('[data-scroll]');
      this.pagesEl = root.querySelector('[data-pages]');
      this.pageInfo = root.querySelector('[data-page-info]');
      this.currentPage = 1;

      root.querySelector('[data-action="prev"]').addEventListener('click', () => this.scrollToPage(this.currentPage - 1));
      root.querySelector('[data-action="next"]').addEventListener('click', () => this.scrollToPage(this.currentPage + 1));
      root.querySelector('[data-action="zoom-in"]').addEventListener('click', () => this.zoomBy(1.15));
      root.querySelector('[data-action="zoom-out"]').addEventListener('click', () => this.zoomBy(1 / 1.15));
      root.querySelector('[data-action="fit"]').addEventListener('click', () => this.setFitMode(true));

      this.scrollEl.addEventListener('scroll', () => this._onScroll(), { passive: true });

      this._resizeObs = new ResizeObserver(() => {
        if (this.fitMode && this.pdfDoc) this.render();
      });
      this._resizeObs.observe(this.scrollEl);
    }

    setLoading(on) {
      this.loadingEl.hidden = !on;
      this.scrollEl.hidden = on;
    }

    async loadFromUrl(url) {
      this.setLoading(true);
      this.pagesEl.innerHTML = '';
      try {
        const pdfjsLib = await ensurePdfJs();
        if (this.pdfDoc) {
          await this.pdfDoc.destroy();
          this.pdfDoc = null;
        }
        const task = pdfjsLib.getDocument(url);
        this.pdfDoc = await task.promise;
        this.fitMode = true;
        this.manualScale = 1;
        await this.render();
        this.setLoading(false);
        this._updateNav();
      } catch (err) {
        this.setLoading(false);
        this.pagesEl.innerHTML = `<p class="pdf-v-error">Nie udało się wyświetlić PDF: ${err.message}</p>`;
        this.scrollEl.hidden = false;
        throw err;
      }
    }

    async loadFromBase64(b64) {
      const raw = atob(b64);
      const buf = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
      const pdfjsLib = await ensurePdfJs();
      this.setLoading(true);
      this.pagesEl.innerHTML = '';
      if (this.pdfDoc) {
        await this.pdfDoc.destroy();
        this.pdfDoc = null;
      }
      this.pdfDoc = await pdfjsLib.getDocument({ data: buf }).promise;
      this.fitMode = true;
      await this.render();
      this.setLoading(false);
      this._updateNav();
    }

    _containerWidth() {
      return Math.max(this.scrollEl.clientWidth - 48, 280);
    }

    _fitScale(page) {
      const vp1 = page.getViewport({ scale: 1 });
      return Math.min(this._containerWidth() / vp1.width, 2.5);
    }

    async _pageScale(page) {
      const fit = this._fitScale(page);
      if (this.fitMode) return fit;
      return Math.min(Math.max(fit * this.manualScale, 0.35), 2.5);
    }

    setFitMode(fit) {
      this.fitMode = fit;
      if (fit) this.manualScale = 1;
      this.render();
    }

    zoomBy(factor) {
      this.fitMode = false;
      this.manualScale = Math.min(Math.max(this.manualScale * factor, 0.5), 2.5);
      this.render();
    }

    async render() {
      if (!this.pdfDoc) return;
      const gen = ++this.renderGen;
      const numPages = this.pdfDoc.numPages;
      this.pagesEl.innerHTML = '';

      for (let n = 1; n <= numPages; n++) {
        if (gen !== this.renderGen) return;
        const page = await this.pdfDoc.getPage(n);
        const scale = await this._pageScale(page);
        const viewport = page.getViewport({ scale });

        const sheet = document.createElement('div');
        sheet.className = 'pdf-v-sheet';
        sheet.dataset.page = String(n);

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        canvas.style.width = `${viewport.width}px`;
        canvas.style.height = `${viewport.height}px`;

        sheet.appendChild(canvas);
        this.pagesEl.appendChild(sheet);

        await page.render({ canvasContext: ctx, viewport }).promise;
      }

      this._updateNav();
      this.scrollToPage(1, false);
    }

    _updateNav() {
      const n = this.pdfDoc?.numPages || 0;
      const prev = this.root.querySelector('[data-action="prev"]');
      const next = this.root.querySelector('[data-action="next"]');
      if (n <= 1) {
        this.pageInfo.textContent = n === 1 ? '1 strona' : '—';
        prev.disabled = true;
        next.disabled = true;
        return;
      }
      this.pageInfo.textContent = `${this.currentPage} / ${n}`;
      prev.disabled = this.currentPage <= 1;
      next.disabled = this.currentPage >= n;
    }

    scrollToPage(num, smooth = true) {
      if (!this.pdfDoc) return;
      const n = Math.max(1, Math.min(num, this.pdfDoc.numPages));
      this.currentPage = n;
      const sheet = this.pagesEl.querySelector(`[data-page="${n}"]`);
      if (sheet) {
        sheet.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto', block: 'start' });
      }
      this._updateNav();
    }

    _onScroll() {
      if (!this.pdfDoc) return;
      const sheets = [...this.pagesEl.querySelectorAll('.pdf-v-sheet')];
      const top = this.scrollEl.scrollTop + 80;
      let active = 1;
      for (const s of sheets) {
        if (s.offsetTop <= top) active = parseInt(s.dataset.page, 10);
      }
      if (active !== this.currentPage) {
        this.currentPage = active;
        this._updateNav();
      }
    }

    destroy() {
      if (this._resizeObs) this._resizeObs.disconnect();
      if (this.pdfDoc) this.pdfDoc.destroy();
      this.pdfDoc = null;
    }
  }

  global.PdfViewer = PdfViewer;
  global.createPdfViewer = (el) => new PdfViewer(el);
})(window);
