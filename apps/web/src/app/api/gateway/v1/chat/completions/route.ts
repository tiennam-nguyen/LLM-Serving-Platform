import { relay } from "../../../../../../lib/gateway/relay";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export function POST(request: Request) { return relay(request, "/v1/chat/completions"); }
