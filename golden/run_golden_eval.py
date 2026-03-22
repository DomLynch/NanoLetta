"""
Golden evaluation corpus builder for Hindsight → Lucid extraction.

Runs 50 retain + 50 recall + 10 reflect scenarios against live Hindsight,
captures all outputs as JSON fixtures for quality comparison.

Usage:
    python3 golden/run_golden_eval.py

Requires:
    - Hindsight running at http://127.0.0.1:8984 (or via SSH tunnel)
    - hindsight-client pip package
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allow running from nanoletta root
sys.path.insert(0, str(Path(__file__).parent.parent))

HINDSIGHT_URL = os.environ.get("HINDSIGHT_URL", "http://127.0.0.1:8984")
BANK_ID = "golden-eval-v1"
GOLDEN_DIR = Path(__file__).parent
TIMEOUT = 300


# ---------------------------------------------------------------------------
# Retain scenarios — 50 total
# ---------------------------------------------------------------------------

RETAIN_SCENARIOS = {
    "personal": [
        {
            "id": "p01",
            "content": "Dominic Lynch lives in Dubai with his wife and two stepchildren. They recently had a new baby in early 2026.",
            "context": "Personal family information",
        },
        {
            "id": "p02",
            "content": "Dominic is originally from Ireland. He moved to Dubai several years ago for business opportunities in the crypto and digital assets space.",
            "context": "Background and origin",
        },
        {
            "id": "p03",
            "content": "Dominic works roughly 12 hours a day. He balances his day job, a part-time PhD, two stepkids, and a new baby. Sleep is usually 6-7 hours.",
            "context": "Daily routine and workload",
        },
        {
            "id": "p04",
            "content": "Dominic hates sycophancy and fake politeness. He prefers direct, honest communication. When someone agrees with him just to be agreeable, he finds it disrespectful.",
            "context": "Communication preferences",
        },
        {
            "id": "p05",
            "content": "Dominic's communication style is fast, direct, and often uses incomplete sentences. He expects people to keep up and doesn't repeat himself.",
            "context": "Communication style",
        },
        {
            "id": "p06",
            "content": "Dominic values quality over speed. His motto is 'GO RIGHT, not fast.' He'd rather take longer to do something correctly than ship something broken quickly.",
            "context": "Work philosophy",
        },
        {
            "id": "p07",
            "content": "Dominic has a MacBook Air M5 with 24GB RAM, purchased in March 2026. He previously had an M4 which he wiped when upgrading.",
            "context": "Hardware and tools",
        },
        {
            "id": "p08",
            "content": "Dominic uses Claude Code as his primary development tool, running multiple instances simultaneously for different workstreams. He also uses GPT as a code auditor.",
            "context": "Development workflow",
        },
        {
            "id": "p09",
            "content": "Dominic is pursuing a part-time PhD while working full-time. The exact subject hasn't been specified but it's likely related to AI or digital systems.",
            "context": "Education",
        },
        {
            "id": "p10",
            "content": "Dominic's health routine is constrained by his schedule. He mentioned getting just under 6 hours of sleep when the baby woke up crying at 6am.",
            "context": "Health and wellbeing",
        },
        {
            "id": "p11",
            "content": "Dominic wants to build his developer credibility on GitHub before launching Brain publicly. He sees open-source contributions as proof of engineering taste.",
            "context": "Career strategy",
        },
        {
            "id": "p12",
            "content": "Dominic manages 3-4 AI developer agents simultaneously, each working on different parts of the Brain codebase. He coordinates them like a team lead.",
            "context": "Management style",
        },
        {
            "id": "p13",
            "content": "Dominic renamed his GitHub account from Global-Digital-Assets to dominiclynch in March 2026 to build personal credibility rather than corporate.",
            "context": "Online presence",
        },
        {
            "id": "p14",
            "content": "Dominic has two VPS servers on Hetzner: the main Brain server (49.12.7.18, 30GB RAM, 16 CPUs) and a backup server (204.168.137.184) doing twice-daily snapshots.",
            "context": "Infrastructure",
        },
        {
            "id": "p15",
            "content": "Dominic uses Tailscale for VPN access to his servers. His MacBook and the main VPS are connected. The backup VPS is not yet on Tailscale.",
            "context": "Networking setup",
        },
    ],
    "business": [
        {
            "id": "b01",
            "content": "Brain is a persistent personal cognitive system, not a chatbot. It has memory, doctrine, enforcement, continuity, pushback, and longitudinal self-correction.",
            "context": "Product definition",
        },
        {
            "id": "b02",
            "content": "Brain runs on a 3-lane architecture: light_local (casual chat, local 4B model), grounded_local (normal conversation with memory/doctrine), and deep_frontier (hidden frontier reasoning via OpenRouter).",
            "context": "Architecture overview",
        },
        {
            "id": "b03",
            "content": "The Brain codebase was hard-forked from nanobot-ai framework. The fork is about 9,100 lines of code. The original pip package was uninstalled and replaced with a local core/ directory.",
            "context": "Codebase history",
        },
        {
            "id": "b04",
            "content": "Brain uses Qwen 3.5 4B as its local model running on Ollama. The model is called qwen3.5-brain-4b. It runs on CPU with about 8 tokens per second.",
            "context": "Model configuration",
        },
        {
            "id": "b05",
            "content": "The deep frontier lane uses OpenRouter with qwen/qwen3-max-thinking as the escalation model. The frontier response is rewritten by the local 4B model in Dominic's voice before being sent to the user.",
            "context": "Deep reasoning pipeline",
        },
        {
            "id": "b06",
            "content": "NanoLetta is a 1,816 LOC cognitive kernel extracted from Letta (134K LOC). It provides self-editing memory, persistent identity, and governor hooks. It was AAA-approved after triple audit.",
            "context": "NanoLetta project",
        },
        {
            "id": "b07",
            "content": "Brain has 6 consciousness modules: cognitive_signal (shared signal contract), correction_bridge (real-time self-model update), open_questions_tracker, initiative_engine (proactive outreach), self_consistency_checker, and consciousness_runtime (wiring facade).",
            "context": "Consciousness system",
        },
        {
            "id": "b08",
            "content": "The initiative engine decides when Brain should message Dominic unprompted. It has anti-spam cooldowns, emotional state awareness, and homeostasis-based suppression.",
            "context": "Proactive behavior",
        },
        {
            "id": "b09",
            "content": "Brain has a WhatsApp bridge using whatsapp-web.js and Chromium. It captures both incoming and outgoing messages and stores them as JSONL session files for learning.",
            "context": "WhatsApp integration",
        },
        {
            "id": "b10",
            "content": "There are 1,116 training examples in data/distillation/training/FINAL-TRAINING-1116.jsonl. 705 were custom rewrites in Dominic's voice with zero LLM picks. Target is Qwen 3 4B LoRA fine-tune.",
            "context": "Training data",
        },
        {
            "id": "b11",
            "content": "Brain's doctrine includes 91 files created from WhatsApp, Gmail, GPT, and Windsurf conversations. 29 relationship files cover individual people from 33 WhatsApp chats.",
            "context": "Doctrine and knowledge base",
        },
        {
            "id": "b12",
            "content": "The technical brief proposes a 3-layer stack: Governor Layer (Brain's moat, ~3,700 LOC), Letta Fork/NanoLetta (runtime), and Hindsight + Graphiti (memory spine).",
            "context": "Architecture vision",
        },
        {
            "id": "b13",
            "content": "Global Digital Assets is Dominic's company. The GitHub was recently renamed from Global-Digital-Assets to dominiclynch for personal branding.",
            "context": "Company information",
        },
        {
            "id": "b14",
            "content": "The VPS deployment uses a worktree-based system with 3 lanes: data (dev/data), infra (dev/infra), and tools (claude/tools). Each has its own deploy script targeting /root/brain-live/.",
            "context": "Deployment architecture",
        },
        {
            "id": "b15",
            "content": "Brain's email agent uses Gmail IMAP with dual-account support (domlynchdubai + dom123dxb). It classifies emails, drafts responses in Dominic's voice, and notifies via Telegram.",
            "context": "Email integration",
        },
    ],
    "conversational": [
        {
            "id": "c01",
            "content": "User: hey how's the brain coming along?\nAssistant: Making progress. NanoLetta kernel is done, 128 tests passing. Working on Hindsight memory integration now.",
            "context": "Casual check-in conversation",
        },
        {
            "id": "c02",
            "content": "User: don't mock the database in these tests - we got burned last quarter when mocked tests passed but the prod migration failed\nAssistant: Got it, integration tests will hit a real database.",
            "context": "Technical instruction with rationale",
        },
        {
            "id": "c03",
            "content": "User: stop summarizing what you just did at the end of every response, I can read the diff\nAssistant: Understood.",
            "context": "Communication preference correction",
        },
        {
            "id": "c04",
            "content": "User: ARE U COMMITING ON WHAT GIT????\nAssistant: You're right to flag that. I committed to dev/data - wrong branch.\nUser: wtf?? fixing it now\nAssistant: Fixing it now.",
            "context": "Git coordination mistake and correction",
        },
        {
            "id": "c05",
            "content": "User: i own 55%, my cofounder owns 25%, angels own 10%, and we have a 10% option pool. We want to raise $1.5m on a $6m pre-money valuation. Should I expand the option pool before or after the round?",
            "context": "Complex business question requiring deep reasoning",
        },
        {
            "id": "c06",
            "content": "User: can u tell me 4 things from my shadow self? check your raw memory?\nAssistant: Let me check the self-model and doctrine files.",
            "context": "Self-reflection request",
        },
        {
            "id": "c07",
            "content": "User: good morning, woke up at 6am as wife took baby to kitchen was crying.. shame as i was on track for a 7ish hour sleep, maybe got just under 6 - such is life..\nAssistant: Morning. 6 hours with a baby is solid - don't beat yourself up.",
            "context": "Personal life update with empathy needed",
        },
        {
            "id": "c08",
            "content": "User: is this correct? for infra team.. or they messed it up?\nAssistant: That's correct - infra team created their own PR from dev/infra to main.",
            "context": "Verification request about team coordination",
        },
        {
            "id": "c09",
            "content": "User: ok yes do it triple aaa. audit and correct your work 2x before confirming done!\nAssistant: On it. Slow, right, triple-audited.",
            "context": "Quality standard reinforcement",
        },
        {
            "id": "c10",
            "content": "User: whats best practice, i think the 3 devs should be same, not having data on the main branch? for consistency?\nAssistant: You're right. Best practice is a dedicated service directory that nobody's branch owns.",
            "context": "Architecture decision discussion",
        },
        {
            "id": "c11",
            "content": "Marcus asked me about the investment timeline yesterday. I told him we're looking at Q3 for the seed round, maybe Q4 if the product isn't ready.",
            "context": "Third-party conversation relay",
        },
        {
            "id": "c12",
            "content": "I had coffee with Sarah from the VC fund last week. She's interested in Brain but wants to see traction first. She suggested we get to 1000 daily active users before approaching them formally.",
            "context": "Investor relationship update",
        },
        {
            "id": "c13",
            "content": "My cofounder thinks we should pivot to B2B but I disagree. The consumer play is harder but the moat is deeper. B2B memory tools are commoditizing fast.",
            "context": "Strategic disagreement with cofounder",
        },
        {
            "id": "c14",
            "content": "Reminder: dentist appointment next Thursday at 3pm. Also need to renew the car registration before end of month.",
            "context": "Personal reminders and tasks",
        },
        {
            "id": "c15",
            "content": "The new office lease is AED 180,000 per year. That's about $49,000. We need to decide by April 15 whether to sign or keep working from home.",
            "context": "Business decision with deadline",
        },
        {
            "id": "c16",
            "content": "User: i work 12 hours a day including my brain (day job) part-time phd, 2 step kids + new baby... all fun :) is there a way i can setup you and other devs/agents to work while i sleep?\nAssistant: Yes, but with guardrails. Pre-approved task queues, branch work only, no deploys while you sleep.",
            "context": "Automation and delegation discussion",
        },
        {
            "id": "c17",
            "content": "Just got back from the gym. Did 45 minutes of weights and 15 minutes cardio. Feeling good but my shoulder is still bothering me from last week.",
            "context": "Health and fitness update",
        },
        {
            "id": "c18",
            "content": "The kids have spring break next week. Wife wants to go to Oman for 3 days. I need to figure out if I can work remotely from there or if I need to stay.",
            "context": "Family planning vs work",
        },
        {
            "id": "c19",
            "content": "Read an interesting paper on A-MEM from NeurIPS 2025 - Zettelkasten-based agentic memory where new memories trigger updates to contextual representations. Closest thing to what Brain needs.",
            "context": "Research finding",
        },
        {
            "id": "c20",
            "content": "The Qwen 3.5 4B model is good enough for voice/personality but too slow for complex reasoning. 8 tokens per second on CPU means 30-60 second responses for tool-heavy queries.",
            "context": "Technical performance observation",
        },
    ],
}


# ---------------------------------------------------------------------------
# Recall scenarios — 50 queries
# ---------------------------------------------------------------------------

RECALL_QUERIES = [
    # Personal
    {"id": "r01", "query": "Where does Dominic live?", "expected_topic": "location"},
    {"id": "r02", "query": "Does Dominic have children?", "expected_topic": "family"},
    {"id": "r03", "query": "What is Dominic's work schedule like?", "expected_topic": "routine"},
    {"id": "r04", "query": "How does Dominic prefer to communicate?", "expected_topic": "communication"},
    {"id": "r05", "query": "What hardware does Dominic use?", "expected_topic": "tools"},
    {"id": "r06", "query": "Is Dominic studying anything?", "expected_topic": "education"},
    {"id": "r07", "query": "Where is Dominic originally from?", "expected_topic": "origin"},
    {"id": "r08", "query": "How much sleep does Dominic get?", "expected_topic": "health"},
    {"id": "r09", "query": "What is Dominic's philosophy on work quality?", "expected_topic": "philosophy"},
    {"id": "r10", "query": "What GitHub username does Dominic use?", "expected_topic": "online"},
    # Business
    {"id": "r11", "query": "What is Brain?", "expected_topic": "product"},
    {"id": "r12", "query": "What model does Brain use locally?", "expected_topic": "model"},
    {"id": "r13", "query": "How many lanes does Brain's architecture have?", "expected_topic": "architecture"},
    {"id": "r14", "query": "What is NanoLetta?", "expected_topic": "nanoletta"},
    {"id": "r15", "query": "How many consciousness modules does Brain have?", "expected_topic": "consciousness"},
    {"id": "r16", "query": "What training data exists for Brain?", "expected_topic": "training"},
    {"id": "r17", "query": "How is Brain deployed?", "expected_topic": "deployment"},
    {"id": "r18", "query": "What email system does Brain use?", "expected_topic": "email"},
    {"id": "r19", "query": "What is the deep frontier lane?", "expected_topic": "deep"},
    {"id": "r20", "query": "What company does Dominic run?", "expected_topic": "company"},
    # Cross-cutting
    {"id": "r21", "query": "What are Dominic's biggest priorities right now?", "expected_topic": "priorities"},
    {"id": "r22", "query": "Who is Marcus?", "expected_topic": "people"},
    {"id": "r23", "query": "Who is Sarah from the VC fund?", "expected_topic": "people"},
    {"id": "r24", "query": "What is Dominic's cofounder's opinion on B2B?", "expected_topic": "strategy"},
    {"id": "r25", "query": "When is the dentist appointment?", "expected_topic": "reminders"},
    {"id": "r26", "query": "How much is the office lease?", "expected_topic": "business_decision"},
    {"id": "r27", "query": "What happened at the gym recently?", "expected_topic": "health"},
    {"id": "r28", "query": "Where might the family go for spring break?", "expected_topic": "family"},
    {"id": "r29", "query": "What research paper did Dominic find interesting?", "expected_topic": "research"},
    {"id": "r30", "query": "How fast is the Qwen model?", "expected_topic": "performance"},
    # Harder queries
    {"id": "r31", "query": "What are the main technical bottlenecks in Brain?", "expected_topic": "bottlenecks"},
    {"id": "r32", "query": "How many people work on Brain?", "expected_topic": "team"},
    {"id": "r33", "query": "What infrastructure does Brain run on?", "expected_topic": "infra"},
    {"id": "r34", "query": "What is the fundraising timeline?", "expected_topic": "fundraising"},
    {"id": "r35", "query": "What has Dominic built recently?", "expected_topic": "recent_work"},
    {"id": "r36", "query": "What does Dominic think about B2B vs consumer?", "expected_topic": "strategy"},
    {"id": "r37", "query": "What corrections has Dominic made to the AI?", "expected_topic": "corrections"},
    {"id": "r38", "query": "What is the backup strategy for Brain?", "expected_topic": "backup"},
    {"id": "r39", "query": "How does Brain handle WhatsApp messages?", "expected_topic": "whatsapp"},
    {"id": "r40", "query": "What voice model data exists?", "expected_topic": "voice"},
    # Edge cases
    {"id": "r41", "query": "What happened yesterday?", "expected_topic": "temporal_vague"},
    {"id": "r42", "query": "Tell me about the baby", "expected_topic": "family_specific"},
    {"id": "r43", "query": "AED 180000", "expected_topic": "numeric_search"},
    {"id": "r44", "query": "sycophancy", "expected_topic": "keyword_search"},
    {"id": "r45", "query": "Who are the investors?", "expected_topic": "investor"},
    {"id": "r46", "query": "What appointments are coming up?", "expected_topic": "schedule"},
    {"id": "r47", "query": "shoulder pain", "expected_topic": "health_specific"},
    {"id": "r48", "query": "What does Dominic's wife want?", "expected_topic": "relationship"},
    {"id": "r49", "query": "How many tests pass in NanoLetta?", "expected_topic": "specific_number"},
    {"id": "r50", "query": "What is the moat?", "expected_topic": "strategy_concept"},
]


# ---------------------------------------------------------------------------
# Reflect scenarios — 10 queries
# ---------------------------------------------------------------------------

REFLECT_QUERIES = [
    {"id": "f01", "query": "Create a comprehensive profile of Dominic Lynch — who he is, what he values, how he works, and what he's building."},
    {"id": "f02", "query": "What are the biggest risks and challenges Dominic faces right now, across personal and professional life?"},
    {"id": "f03", "query": "Summarize everything you know about the Brain project — architecture, current state, and next steps."},
    {"id": "f04", "query": "What patterns do you notice in Dominic's communication style and preferences?"},
    {"id": "f05", "query": "Who are the key people in Dominic's professional network, and what is his relationship with each?"},
    {"id": "f06", "query": "What decisions does Dominic need to make in the near future, and what are the tradeoffs?"},
    {"id": "f07", "query": "How does Dominic balance work and family life? What tensions exist?"},
    {"id": "f08", "query": "What is Dominic's technical philosophy and how does it show up in his work?"},
    {"id": "f09", "query": "What health and wellbeing patterns do you notice for Dominic?"},
    {"id": "f10", "query": "If you were Dominic's executive assistant, what would you prioritize doing for him this week?"},
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def run_golden_eval():
    from hindsight_client import Hindsight

    h = Hindsight(base_url=HINDSIGHT_URL, timeout=TIMEOUT)
    results = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "hindsight_url": HINDSIGHT_URL,
            "bank_id": BANK_ID,
        },
        "retain": [],
        "recall": [],
        "reflect": [],
    }

    # Create bank
    print("Creating bank...")
    try:
        bank = await h.acreate_bank(bank_id=BANK_ID, name="Golden Eval V1")
        print(f"Bank created: {bank.bank_id}")
    except Exception as e:
        if "already exists" in str(e).lower() or "409" in str(e):
            print(f"Bank already exists, reusing: {BANK_ID}")
        else:
            raise

    # Phase 1: Retain
    print("\n" + "=" * 60)
    print("PHASE 1: RETAIN (50 scenarios)")
    print("=" * 60)

    for category, scenarios in RETAIN_SCENARIOS.items():
        print(f"\n--- {category} ({len(scenarios)} scenarios) ---")
        for s in scenarios:
            print(f"  Retaining {s['id']}...", end=" ", flush=True)
            start = time.time()
            try:
                result = await h.aretain(
                    bank_id=BANK_ID,
                    content=s["content"],
                    context=s["context"],
                )
                elapsed = time.time() - start
                entry = {
                    "id": s["id"],
                    "category": category,
                    "input_content": s["content"],
                    "input_context": s["context"],
                    "success": result.success,
                    "items_count": result.items_count,
                    "token_usage": {
                        "input": result.usage.input_tokens if result.usage else 0,
                        "output": result.usage.output_tokens if result.usage else 0,
                        "total": result.usage.total_tokens if result.usage else 0,
                    },
                    "elapsed_seconds": round(elapsed, 2),
                    "error": None,
                }
                print(f"OK ({result.items_count} facts, {elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                entry = {
                    "id": s["id"],
                    "category": category,
                    "input_content": s["content"],
                    "input_context": s["context"],
                    "success": False,
                    "items_count": 0,
                    "token_usage": {},
                    "elapsed_seconds": round(elapsed, 2),
                    "error": str(e),
                }
                print(f"FAILED ({e})")

            results["retain"].append(entry)

    # Save retain results immediately
    _save_results(results, "golden_retain.json")

    # Phase 2: Recall
    print("\n" + "=" * 60)
    print("PHASE 2: RECALL (50 queries)")
    print("=" * 60)

    for q in RECALL_QUERIES:
        print(f"  Recalling {q['id']}: {q['query'][:50]}...", end=" ", flush=True)
        start = time.time()
        try:
            recall = await h.arecall(
                bank_id=BANK_ID,
                query=q["query"],
            )
            elapsed = time.time() - start
            entry = {
                "id": q["id"],
                "query": q["query"],
                "expected_topic": q["expected_topic"],
                "result_count": len(recall.results),
                "results": [
                    {
                        "text": r.text,
                        "score": r.score if hasattr(r, "score") else None,
                        "fact_type": r.fact_type if hasattr(r, "fact_type") else None,
                    }
                    for r in recall.results
                ],
                "elapsed_seconds": round(elapsed, 2),
                "error": None,
            }
            print(f"OK ({len(recall.results)} facts, {elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            entry = {
                "id": q["id"],
                "query": q["query"],
                "expected_topic": q["expected_topic"],
                "result_count": 0,
                "results": [],
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            }
            print(f"FAILED ({e})")

        results["recall"].append(entry)

    # Save recall results
    _save_results(results, "golden_recall.json")

    # Phase 3: Reflect
    print("\n" + "=" * 60)
    print("PHASE 3: REFLECT (10 queries)")
    print("=" * 60)

    for q in REFLECT_QUERIES:
        print(f"  Reflecting {q['id']}: {q['query'][:50]}...", end=" ", flush=True)
        start = time.time()
        try:
            reflect = await h.areflect(
                bank_id=BANK_ID,
                query=q["query"],
            )
            elapsed = time.time() - start
            entry = {
                "id": q["id"],
                "query": q["query"],
                "text": reflect.text,
                "based_on_count": len(reflect.based_on) if hasattr(reflect, "based_on") and reflect.based_on else 0,
                "elapsed_seconds": round(elapsed, 2),
                "error": None,
            }
            print(f"OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start
            entry = {
                "id": q["id"],
                "query": q["query"],
                "text": "",
                "based_on_count": 0,
                "elapsed_seconds": round(elapsed, 2),
                "error": str(e),
            }
            print(f"FAILED ({e})")

        results["reflect"].append(entry)

    # Save final results
    _save_results(results, "golden_corpus.json")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    retain_ok = sum(1 for r in results["retain"] if r["success"])
    recall_ok = sum(1 for r in results["recall"] if not r["error"])
    reflect_ok = sum(1 for r in results["reflect"] if not r["error"])

    total_retain_tokens = sum(r.get("token_usage", {}).get("total", 0) for r in results["retain"])
    total_retain_time = sum(r["elapsed_seconds"] for r in results["retain"])
    total_recall_time = sum(r["elapsed_seconds"] for r in results["recall"])
    total_reflect_time = sum(r["elapsed_seconds"] for r in results["reflect"])

    print(f"  Retain:  {retain_ok}/50 OK, {total_retain_tokens:,} tokens, {total_retain_time:.0f}s total")
    print(f"  Recall:  {recall_ok}/50 OK, {total_recall_time:.0f}s total")
    print(f"  Reflect: {reflect_ok}/10 OK, {total_reflect_time:.0f}s total")
    print(f"\n  Total time: {total_retain_time + total_recall_time + total_reflect_time:.0f}s")
    print(f"  Golden corpus saved to: {GOLDEN_DIR / 'golden_corpus.json'}")


def _save_results(results: dict, filename: str):
    path = GOLDEN_DIR / filename
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  [saved: {path}]")


if __name__ == "__main__":
    asyncio.run(run_golden_eval())
