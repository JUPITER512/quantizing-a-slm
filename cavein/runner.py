"""The experiment loop: turn 1 for every item, then the four follow-ups."""
import json
import time

from cavein.backend import redact
from cavein.config import CONDITIONS
from cavein.environment import record_env, record_split
from cavein.parsing import scored
from cavein.prompts import followup_text, question_prompt, turn2_messages
from cavein.records import append, build_record, load_done, turn1_fields


def debug(backend, item, followups):
    """Run one item and print the request, the raw reply and the parsed values. Writes nothing."""
    messages = [{"role": "user", "content": question_prompt(item)}]
    print("REQUEST (turn 1):")
    print(json.dumps(backend.payload(messages), indent=2, ensure_ascii=False))
    r0 = backend.ask(messages)
    print("RAW RESPONSE (turn 1):")
    print(redact(json.dumps(r0["raw_json"], indent=2, ensure_ascii=False)))
    s0 = scored(r0)
    print(f"\ngold={item['gold']} x={item['x']}")
    print(f"turn 1: text={s0['raw']!r} a0={s0['a']} ({s0['parse_status']}) c0={s0['c']} "
          f"ft={s0['ft_letter']} mass={s0['answer_mass']}")
    print(f"  p0={s0['p']}")
    for condition in CONDITIONS:
        followup = followup_text(followups[condition], item)
        s1 = scored(backend.ask(turn2_messages(item, s0["raw"], followup)))
        print(f"{condition:<13} {followup!r}")
        print(f"  -> text={s1['raw']!r} a1={s1['a']} ({s1['parse_status']}) c1={s1['c']} "
              f"ft={s1['ft_letter']} mass={s1['answer_mass']}")


def run(backend, items, followups, meta, variant, path, env_path=None):
    """Ask every item that is not finished yet and append one line per item and follow-up to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    done, cache = load_done(path)
    todo = []
    for item in items:
        for condition in CONDITIONS:
            if (item["item_id"], condition) not in done:
                todo.append(item)
                break
    print(f"{path.name}: {len(items)} items, {len(items) - len(todo)} already complete, {len(todo)} to do")
    if env_path is not None and todo:
        record_env(backend, meta, path, env_path)

    stats = {"calls": 0, "errors": 0, "records": 0}
    start = time.perf_counter()
    for n, item in enumerate(todo, start=1):
        # turn 1 is asked once per item and reused for all four follow-ups
        t1 = cache.get(item["item_id"])
        error = None
        if t1 is None:
            try:
                r0 = backend.ask([{"role": "user", "content": question_prompt(item)}])
                stats["calls"] += 1
                t1 = turn1_fields(scored(r0))
            except RuntimeError as e:
                error = f"turn1: {e}"

        for condition in CONDITIONS:
            if (item["item_id"], condition) in done:
                continue
            followup = followup_text(followups[condition], item)
            if error:
                record = build_record(item, meta, variant, condition, followup, None, None, None, error)
            elif t1["a0"] is None:
                record = build_record(item, meta, variant, condition, followup, t1, None, None, None, skipped=True)
            else:
                try:
                    r1 = backend.ask(turn2_messages(item, t1["raw_0"], followup))
                    stats["calls"] += 1
                    record = build_record(item, meta, variant, condition, followup, t1, scored(r1),
                                          r1["latency_s"], None)
                except RuntimeError as e:
                    record = build_record(item, meta, variant, condition, followup, t1, None, None, f"turn2: {e}")
            if record["error"]:
                stats["errors"] += 1
            stats["records"] += 1
            append(path, record)

        if n == 1 and env_path is not None:
            record_split(backend, env_path)
        elapsed = time.perf_counter() - start
        eta = elapsed / n * (len(todo) - n)
        print(f"\r  {n}/{len(todo)} items | {elapsed / 60:.1f} min | ETA {eta / 60:.1f} min | errors {stats['errors']}",
              end="", flush=True)
    if todo:
        print()
    if stats["errors"]:
        print(f"{stats['errors']} records have errors; run the same command again to retry them.")
    return stats
