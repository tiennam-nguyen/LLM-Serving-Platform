import type { ReactNode } from "react";

export function Plate({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`plate chamfer ${className}`}><div className="face">
    <i className="screw tl" aria-hidden="true" /><i className="screw tr" aria-hidden="true" />
    {children}
  </div></section>;
}
