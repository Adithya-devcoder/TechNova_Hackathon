import { ArrowRight } from "lucide-react";

function GestureGraphic() {
  return (
    <svg viewBox="0 0 400 320" className="h-auto w-full" aria-hidden="true">
      <g fill="none" strokeLinecap="round">
        {Array.from({ length: 7 }).map((_, i) => (
          <path key={i}
            d={`M20 ${250 - i * 22} C 110 ${250 - i * 22}, 120 ${80 + i * 14}, 200 ${160} S 300 ${240 - i * 26}, 380 ${70 + i * 24}`}
            stroke="var(--color-border)" strokeWidth="1" />
        ))}
        <path d="M20 184 C 110 184, 120 122, 200 160 S 300 162, 380 142" stroke="var(--color-primary)" strokeWidth="1.6" />
      </g>
      <circle cx="20" cy="184" r="5" fill="var(--color-foreground)" />
      <circle cx="200" cy="160" r="3.5" fill="var(--color-primary)" />
      <circle cx="380" cy="142" r="5" fill="none" stroke="var(--color-primary)" strokeWidth="1.6" />
      <text x="20" y="300" fill="var(--color-muted-foreground)" fontSize="10" letterSpacing="2" fontFamily="Manrope">GESTURE</text>
      <text x="380" y="300" textAnchor="end" fill="var(--color-muted-foreground)" fontSize="10" letterSpacing="2" fontFamily="Manrope">LANGUAGE</text>
    </svg>
  );
}

export function Hero() {
  return (
    <section id="top" className="grid items-end gap-10 border-b py-12 md:grid-cols-[1.2fr_1fr] md:py-16">
      <div>
        <p className="eyebrow">Accessibility / Communication technology</p>
        <h1 className="mt-6 font-serif text-6xl leading-[0.92] tracking-tight sm:text-7xl lg:text-[7.5rem]">
          Beyond<br />words.<br />
          <span className="text-muted-foreground">Beyond</span><br />
          <span className="italic text-primary">barriers.</span>
        </h1>
      </div>
      <div className="flex flex-col gap-8">
        <GestureGraphic />
        <p className="max-w-sm text-lg leading-relaxed text-muted-foreground">
          Turning continuous signs into readable language, one gesture at a time.
        </p>
        <a href="#translate" className="btn btn-primary group h-12 w-fit px-6">
          Start translating <ArrowRight className="size-4 transition-transform duration-200 group-hover:translate-x-1" />
        </a>
      </div>
    </section>
  );
}
