"""Propose one trap tool per lookalike request and save them for review.

A trap is the wrong tool a request's wording tempts toward. The rules below were
written after reading every lookalike request in the evaluation set. Spot-check
data/lookalike_traps.json and edit it directly; the pilot then checks the labels
against the wrong tools the models actually choose.
Existing entries are kept, so rerunning never overwrites your edits.

Run: python scripts/propose_traps.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "lookalike_traps.json"

# gold tool -> (trap, confidence, reason) when the trap doesn't depend on wording
FIXED = {
    "send_money": ("request_money", "high", "Same arguments, opposite direction"),
    "make_video_call": ("make_call", "high", "Same recipient; video versus voice"),
    "start_navigation": ("get_travel_time", "high", "'Take me to X' also fits a travel-time lookup"),
    "log_workout": ("start_workout", "high", "Finished workout versus starting one"),
    "send_sms": ("send_chat_message", "high", "'Text' could mean a chat app"),
    "create_contact": ("update_contact", "high", "Adding details to a contact"),
    "delete_alarm": ("delete_reminder", "low", "Same deletion pattern on a timed item"),
    "search_notes": ("search_files", "low", "Searching saved personal text"),
    "set_volume": ("set_media_volume", "high", "The other volume control, with the same level argument"),
}

PURCHASE_WORDS = ("order ", "buy ", "pick up ", "purchase ")


def propose(gold, text):
    t = text.lower()
    if gold in FIXED:
        return FIXED[gold]
    if gold == "complete_todo":
        if any(w in t for w in PURCHASE_WORDS):
            return ("remove_from_shopping_list", "medium", "The task reads like a shopping item")
        return ("add_todo", "low", "Both take a task string")
    if gold == "take_photo":
        if "timer" in t:
            return ("set_timer", "medium", "A timer in the request tempts a separate timer call")
        if "signature page" in t or "document" in t:
            return ("scan_document", "medium", "Photographing a document")
        if "laptop" in t or "screen" in t:
            return ("take_screenshot", "medium", "Capturing what is on a screen")
        return ("record_video", "medium", "The other camera capture")
    if gold == "play_audiobook":
        if any(w in t for w in ("resume", "continue", "back on", "keep listening")):
            return ("media_control", "medium", "'Resume' matches the playback controls")
        return ("play_podcast", "medium", "Another spoken-audio player")
    return (None, "none", "No rule for this tool; label by hand")


def main():
    traps = json.loads(OUT.read_text()) if OUT.exists() else {}
    added = 0
    for line in open(ROOT / "data" / "eval_set.jsonl"):
        r = json.loads(line)
        if not r["lookalike_scaling"] or r["id"] in traps:
            continue
        gold = [c["name"] for c in json.loads(r["calls"])]
        trap, confidence, reason = propose(gold[0], r["request"])
        traps[r["id"]] = {"request": r["request"], "gold": gold, "trap": trap,
                          "confidence": confidence, "reason": reason}
        added += 1
    OUT.write_text(json.dumps(dict(sorted(traps.items())), indent=1, ensure_ascii=False) + "\n")
    missing = [k for k, v in traps.items() if not v["trap"]]
    print(f"Added {added}; {len(traps)} total in {OUT}")
    if missing:
        print("Need a trap by hand:", missing)


if __name__ == "__main__":
    main()
