(() => {
  const article = document.querySelector("[data-post-content]");
  const progress = document.querySelector("[data-reading-progress]");

  if (!article) {
    return;
  }

  const headings = [...article.querySelectorAll("h2, h3")];
  const tocPanel = document.querySelector("[data-toc-panel]");
  const toc = document.querySelector("[data-toc]");
  const usedIds = new Map();

  const slugify = (value) =>
    value
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");

  headings.forEach((heading) => {
    const baseId = heading.id || slugify(heading.textContent) || "section";
    const occurrence = usedIds.get(baseId) || 0;
    usedIds.set(baseId, occurrence + 1);
    heading.id = occurrence === 0 ? baseId : `${baseId}-${occurrence + 1}`;
  });

  if (headings.length > 0 && tocPanel && toc) {
    const list = document.createElement("ol");

    headings.forEach((heading) => {
      const item = document.createElement("li");
      const link = document.createElement("a");
      item.dataset.level = heading.tagName.slice(1);
      link.href = `#${heading.id}`;
      link.textContent = heading.textContent;
      item.append(link);
      list.append(item);
    });

    toc.append(list);
    tocPanel.hidden = false;

    if ("IntersectionObserver" in window) {
      const links = new Map(
        [...toc.querySelectorAll("a")].map((link) => [
          link.getAttribute("href").slice(1),
          link,
        ]),
      );

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (!entry.isIntersecting) {
              return;
            }

            links.forEach((link) => link.classList.remove("is-active"));
            links.get(entry.target.id)?.classList.add("is-active");
          });
        },
        { rootMargin: "-15% 0px -70% 0px" },
      );

      headings.forEach((heading) => observer.observe(heading));
    }
  }

  if (progress) {
    let ticking = false;

    const updateProgress = () => {
      const articleTop = article.offsetTop;
      const articleHeight = article.offsetHeight;
      const viewportBottom = window.scrollY + window.innerHeight;
      const distance = Math.max(articleHeight - window.innerHeight, 1);
      const percent = Math.min(
        100,
        Math.max(0, ((viewportBottom - articleTop - window.innerHeight) / distance) * 100),
      );
      progress.style.width = `${percent}%`;
      ticking = false;
    };

    window.addEventListener(
      "scroll",
      () => {
        if (!ticking) {
          window.requestAnimationFrame(updateProgress);
          ticking = true;
        }
      },
      { passive: true },
    );

    updateProgress();
  }
})();
