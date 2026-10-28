import { relay } from "../../../../lib/gateway/relay";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export function GET(request: Request) { return relay(request, "/health"); }
