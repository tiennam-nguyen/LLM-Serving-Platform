import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "LLM Serving Platform / Instrumentation Deck", description: "Streaming inference playground with honest client-observed request instrumentation." };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
