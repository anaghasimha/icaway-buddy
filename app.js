// Keep it simple: just render the modal links (and also fill the inline fallback)
document.addEventListener("DOMContentLoaded", () => {
  const COUNTRIES = [
    { name: "Argentina", slug: "ce_ar" }, { name: "Australia", slug: "ce_au" },
    { name: "Austria", slug: "ce_at" },   { name: "Belgium", slug: "ce_be" },
    { name: "Brazil", slug: "ce_br" },    { name: "Canada", slug: "ce_ca" },
    { name: "Chile", slug: "ce_cl" },     { name: "China", slug: "ce_cn" },
    { name: "Colombia", slug: "ce_co" },  { name: "Czech Republic", slug: "ce_cz" },
    { name: "Denmark", slug: "ce_dk" },   { name: "Egypt", slug: "ce_eg" },
    { name: "Finland", slug: "ce_fi" },   { name: "France", slug: "ce_fr" },
    { name: "Germany", slug: "ce_de" },   { name: "Greece", slug: "ce_gr" },
    { name: "Hong Kong", slug: "ce_hk" }, { name: "Hungary", slug: "ce_hu" },
    { name: "India", slug: "ce_in" },     { name: "Indonesia", slug: "ce_id" },
    { name: "Ireland", slug: "ce_ie" },   { name: "Israel", slug: "ce_il" },
    { name: "Italy", slug: "ce_it" },     { name: "Japan", slug: "ce_jp" },
    { name: "Malaysia", slug: "ce_my" },  { name: "Mexico", slug: "ce_mx" },
    { name: "Netherlands", slug: "ce_nl" },{ name: "New Zealand", slug: "ce_nz" },
    { name: "Norway", slug: "ce_no" },    { name: "Peru", slug: "ce_pe" },
    { name: "Philippines", slug: "ce_ph" },{ name: "Poland", slug: "ce_pl" },
    { name: "Portugal", slug: "ce_pt" },  { name: "Russia", slug: "ce_ru" },
    { name: "Saudi Arabia", slug: "ce_sa" },{ name: "Singapore", slug: "ce_sg" },
    { name: "South Africa", slug: "ce_za" },{ name: "South Korea", slug: "ce_kr" },
    { name: "Spain", slug: "ce_es" },     { name: "Sweden", slug: "ce_se" },
    { name: "Switzerland", slug: "ce_ch" },{ name: "Thailand", slug: "ce_th" },
    { name: "Turkey", slug: "ce_tr" },    { name: "United Arab Emirates", slug: "ce_ae" },
    { name: "United Kingdom", slug: "ce_gb" },{ name: "United States", slug: "ce_us" },
    { name: "Venezuela", slug: "ce_ve" }
  ];

  const openBtn = document.getElementById("openEtiquette");
  const modal = document.getElementById("etiquetteModal");
  const closeBtn = document.getElementById("closeEtiquette");
  const grid = document.getElementById("etiquetteLinks");
  const inlineGrid = document.getElementById("inlineLinks");

  function buildLinks(container) {
    container.innerHTML = "";
    COUNTRIES.forEach(c => {
      const a = document.createElement("a");
      a.href = `http://www.ediplomat.com/np/cultural_etiquette/${c.slug}.htm`;
      a.target = "_blank";
      a.rel = "noopener";
      a.textContent = c.name;
      container.appendChild(a);
    });
  }

  // Build inline fallback (kept hidden unless you want to show it)
  if (inlineGrid) buildLinks(inlineGrid);

  // Open modal
  openBtn?.addEventListener("click", () => {
    buildLinks(grid);
    modal.classList.remove("hidden");
    modal.setAttribute("aria-hidden", "false");
  });

  // Close modal
  const closeModal = () => {
    modal.classList.add("hidden");
    modal.setAttribute("aria-hidden", "true");
  };
  closeBtn?.addEventListener("click", closeModal);
  modal?.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

  // Debug: confirm JS loaded
  console.log(`Loaded ${COUNTRIES.length} etiquette links`);
});
