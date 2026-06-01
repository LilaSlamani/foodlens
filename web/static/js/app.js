/* Initialisation des animations au défilement */
AOS.init({ once: true, duration: 560, easing: 'ease-out-quad', offset: 40 });

/* Scanner de code-barres via la caméra */
async function startScan() {
  const video    = document.getElementById('scan-video');
  const errorEl  = document.querySelector('[x-text="error"]');
  if (!video) return;

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
    video.srcObject = stream;
    video.classList.remove('hidden');

    if (!('BarcodeDetector' in window)) {
      if (errorEl) errorEl.textContent = 'Scanner non supporté. Utilisez Chrome ou Edge.';
      stream.getTracks().forEach(t => t.stop());
      return;
    }

    const detector = new BarcodeDetector({ formats: ['ean_13', 'ean_8', 'code_128', 'qr_code'] });

    /* Boucle de détection image par image */
    const scan = async () => {
      try {
        const codes = await detector.detect(video);
        if (codes.length > 0) {
          stream.getTracks().forEach(t => t.stop());
          video.classList.add('hidden');
          window.location.href = `/produit/${codes[0].rawValue}`;
        } else {
          requestAnimationFrame(scan);
        }
      } catch {
        requestAnimationFrame(scan);
      }
    };
    requestAnimationFrame(scan);

  } catch (err) {
    const msg = err.name === 'NotAllowedError'
      ? 'Accès caméra refusé. Autorisez la caméra dans les paramètres.'
      : "Impossible d'accéder à la caméra.";
    if (errorEl) errorEl.textContent = msg;
  }
}

/* Animation des barres nutritionnelles au chargement */
window.addEventListener('load', () => {
  document.querySelectorAll('[data-width]').forEach(el => {
    setTimeout(() => { el.style.width = el.getAttribute('data-width') + '%'; }, 120);
  });
});

/* Basculer l'affichage d'une section avec rotation de l'icône */
function toggleSection(id) {
  const el   = document.getElementById(id);
  const icon = document.getElementById(id + '-icon');
  if (!el) return;
  el.classList.toggle('hidden');
  if (icon) icon.style.transform = el.classList.contains('hidden') ? '' : 'rotate(180deg)';
}
