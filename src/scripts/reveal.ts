// Scroll-reveal for elements marked with [data-reveal]. Adds "is-visible" when an
// element enters the viewport, which also starts its chart entrance animations.
// The hidden pre-reveal state is only armed (html.reveal-ready) once this runs,
// so content stays visible if JavaScript is unavailable.

const root = document.documentElement;
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const targets = Array.from(document.querySelectorAll<HTMLElement>("[data-reveal]"));

if (reduceMotion || !("IntersectionObserver" in window)) {
  targets.forEach((el) => el.classList.add("is-visible"));
} else if (targets.length > 0) {
  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.12 }
  );
  root.classList.add("reveal-ready");
  targets.forEach((el) => observer.observe(el));
}
