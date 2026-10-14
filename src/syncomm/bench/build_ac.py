"""AC-suite mechanical generator: Countries + Tip Sheets -> SYNC episode JSONL.

Spec: ``the study protocol`` §1.1 (AC Countries/Tip Sheets are the
co-primary, headroom-by-construction tasks; published protocol = Ramesh & Li,
ICML'25, arXiv 2501.14082) and the B0 progress log "REBUILD DECISION": the AC
paper released NO code/data, so the suite is rebuilt fully mechanically
(templates x fixed hand-written word lists x a seeded RNG), scaled from the
published sizes (Countries n=100, Tip Sheets n=70) to n=400 per split, with the
published sizes cited here and in every summary json.

WHY HEADROOM EXISTS BY CONSTRUCTION
-----------------------------------
The receiver's question is unanswerable without the sender's private evidence:

* ac-countries: the sender privately observes "[PERSON] is at [the] [LANDMARK]."
  for a synthetic person and a randomly paired world-famous landmark; the
  receiver is asked "Which country is [PERSON] in?" (4-way MC over the gold
  country + 3 distractor countries drawn from other landmarks' countries).
  The person<->landmark pairing is random, so no world knowledge about the
  person exists; only the private fact plus landmark->country world knowledge
  answers the question.
* ac-tipsheets: the sender privately holds a 3-company tip sheet; each company
  carries a mechanically generated signal profile (revenue direction+pct,
  categorical event, analyst rating) and the sheet states the deterministic
  profile->action decision rule. The receiver is asked the recommended action
  for one named company (BUY/SELL/HOLD + NO DATA distractor). Company names
  are synthetic, profiles are random: the gold is derivable ONLY from the sheet.

EPISODE SCHEMA -- exactly build_bench.py's schema (same keys, same helpers)::

    item_id          "ac-countries/<split>/000123"   (6-digit per-split index)
    benchmark        ac-countries | ac-tipsheets
    split            dev | test   (test written but ONE-TOUCH, Hard Rule 6)
    sender_evidence  the private fact / the full tip-sheet text
    evidence_source  "ac_mechanical_v1"
    question_stem    receiver-visible question (entity name only, no evidence)
    choices          [{"label": "A", "text": ...}, ...] always 4, A-D
    answerKey        one of A B C D  (exactly balanced per split by schedule)
    leakage_flags    build_bench.screen_leakage output + AC-specific screens
    item_sha8        sha256(question + choices + answerKey)[:8]

plus one superset field ``ac_meta`` (generation provenance: person/landmark or
company profiles + rule labels) so that rule correctness and split disjointness
are auditable from the artifact itself without re-running the generator.

LEAKAGE RULES ENFORCED AT BUILD TIME (generator aborts on any violation;
                                      cf. build_bench which logs-only):
* landmark table: no landmark name contains ANY table country's name (raw
  substring; drops e.g. "Great Wall of China"), and no landmark/person/company
  content token collides with any choice-side content token.
* countries: the landmark never appears in the question; the question shares no
  content word with the evidence beyond the person's name; no choice (country)
  string or token occurs in the evidence.
* tipsheets: the question contains no signal word (SIGNAL_WORDS below); the
  question shares no content word with the evidence beyond the asked company's
  name and the two framing tokens {"tip","sheet"}; no choice string occurs in
  the evidence -- the rule text says "buying/selling/holding is advised", never
  the literal BUY/SELL/HOLD, and the neutral analyst rating is "unchanged"
  (NOT the spec's "hold", which would put the HOLD choice verbatim inside every
  third evidence line -- deviation flagged in the module constants).
* both: gold letter exactly balanced A-D per split (100 each at n=400), so the
  letter is not inferable from the question alone.

DISTRACTOR PLAUSIBILITY RULE (countries) -- adversarial review 2026-08-04
-------------------------------------------------------------------------
The dev-lock review found (row-level verified) that uniformly sampled
distractor countries allow GENERIC-TOKEN ANSWERING: a packet that delivers
only the gold landmark's generic head token (e.g. "tower" from "Belem Tower")
often suffices to pick the gold country against random distractors (France vs
Peru vs Egypt). Fix, mechanical and tiered:

  1. shared-token tier: distractor countries are sampled from countries that
     host >= 1 table landmark sharing a generic head token (GENERIC_HEAD_TOKENS:
     tower/temple/palace/bridge/museum/cathedral/castle/fall/square/...) with
     the gold landmark;
  2. same-continent tier: when fewer than 3 such countries exist (or the gold
     landmark has no generic head token -- Uluru, Petra, Stonehenge ...), the
     remainder come from the gold country's continent (CONTINENTS table);
  3. global tier, last resort: only when both pools are exhausted -- reachable
     only for Oceania golds (2 table countries on that continent).

Every distractor's tier is recorded in ``ac_meta.distractors[*].source`` and
aggregated in the summary json (``distractor_rule`` block), so the
shared-token fraction and the fallback rates are auditable per artifact.

DETERMINISM
-----------
One frozen seed (20260805), per-task sub-RNGs derived via sha256 -- two runs of
the same command are byte-identical (tests/test_build_ac.py pins the file
hash). dev and test are generated jointly so the (person,landmark) /
(company,profile) combination sets are DISJOINT across splits by construction.

USAGE::

    PYTHONPATH=src python3 -m syncomm.bench.build_ac \\
        --out-dir data/bench --splits dev test --allow-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

from syncomm.bench.build_bench import (
    LABELS, N_CHOICES, assert_choice_balance, content_words, contains_verbatim,
    item_sha8, leakage_summary, norm_text, screen_leakage, sha256_file, utcnow,
    write_jsonl)

SEED = 20260805
EVIDENCE_SOURCE = "ac_mechanical_v1"
PLAN = ("the study protocol §1.1 (AC co-primary) + B0 log "
        "'REBUILD DECISION' 2026-08-05")

PUBLISHED = {
    "ac-countries": {"published_n": 100,
                     "protocol": "Ramesh & Li, ICML'25 (arXiv 2501.14082), "
                                 "Countries task, mechanical rebuild"},
    "ac-tipsheets": {"published_n": 70,
                     "protocol": "Ramesh & Li, ICML'25 (arXiv 2501.14082), "
                                 "Tip Sheets task, mechanical rebuild"},
}

# ---------------------------------------------------------------------------
# Countries task: hand-written landmark -> country table (verified facts only).
# Inclusion rules (enforced mechanically by validate_pools / the unit tests):
#   * world-famous, single-country landmarks only -- border-straddling or
#     politically ambiguous sites are EXCLUDED by curation (Mount Everest,
#     Niagara Falls, Victoria Falls, Iguazu Falls, Lake Titicaca, Dead Sea,
#     Matterhorn, Taipei 101, St. Peter's Basilica ...);
#   * no landmark whose name contains ANY table country's name -- that is the
#     leakage rule that drops "Great Wall of China", "Panama Canal",
#     "Gateway of India", "Mount Kenya", "Bahrain Fort"; near-demonym cases
#     ("Hungarian Parliament Building") and choice-token collisions
#     ("Empire State Building" vs "United States", "Burj Al Arab" vs
#     "United Arab Emirates") are excluded by curation and pinned by the
#     token-level assert;
#   * third field = whether the evidence sentence uses "the" ("at the Eiffel
#     Tower" vs "at Machu Picchu").
# ---------------------------------------------------------------------------
LANDMARKS = [
    # --- Europe ---
    ("Eiffel Tower", "France", True),
    ("Louvre Museum", "France", True),
    ("Palace of Versailles", "France", True),
    ("Arc de Triomphe", "France", True),
    ("Mont Saint-Michel", "France", False),
    ("Pont du Gard", "France", True),
    ("Chateau de Chambord", "France", False),
    ("Big Ben", "United Kingdom", False),
    ("Tower Bridge", "United Kingdom", True),
    ("Buckingham Palace", "United Kingdom", False),
    ("Stonehenge", "United Kingdom", False),
    ("London Eye", "United Kingdom", True),
    ("Edinburgh Castle", "United Kingdom", False),
    ("Windsor Castle", "United Kingdom", False),
    ("Colosseum", "Italy", True),
    ("Leaning Tower of Pisa", "Italy", True),
    ("Trevi Fountain", "Italy", True),
    ("Rialto Bridge", "Italy", True),
    ("Pompeii", "Italy", False),
    ("Milan Cathedral", "Italy", False),
    ("Uffizi Gallery", "Italy", True),
    ("Sagrada Familia", "Spain", True),
    ("Alhambra", "Spain", True),
    ("Park Guell", "Spain", False),
    ("Seville Cathedral", "Spain", False),
    ("Royal Palace of Madrid", "Spain", True),
    ("Belem Tower", "Portugal", True),
    ("Pena Palace", "Portugal", False),
    ("Brandenburg Gate", "Germany", True),
    ("Neuschwanstein Castle", "Germany", False),
    ("Cologne Cathedral", "Germany", False),
    ("Reichstag Building", "Germany", True),
    ("Anne Frank House", "Netherlands", True),
    ("Rijksmuseum", "Netherlands", True),
    ("Keukenhof Gardens", "Netherlands", True),
    ("Atomium", "Belgium", True),
    ("Manneken Pis", "Belgium", False),
    ("Grand Place", "Belgium", True),
    ("Chapel Bridge", "Switzerland", True),
    ("Jungfraujoch", "Switzerland", False),
    ("Belvedere Palace", "Austria", True),
    ("Hofburg Palace", "Austria", True),
    ("Charles Bridge", "Czech Republic", True),
    ("Prague Castle", "Czech Republic", False),
    ("Wawel Castle", "Poland", False),
    ("Malbork Castle", "Poland", False),
    ("Fisherman's Bastion", "Hungary", True),
    ("Buda Castle", "Hungary", False),
    ("Parthenon", "Greece", True),
    ("Acropolis of Athens", "Greece", True),
    ("Santorini", "Greece", False),
    ("Meteora", "Greece", False),
    ("Hagia Sophia", "Turkey", False),
    ("Blue Mosque", "Turkey", True),
    ("Cappadocia", "Turkey", False),
    ("Pamukkale", "Turkey", False),
    ("Red Square", "Russia", False),
    ("Saint Basil's Cathedral", "Russia", False),
    ("Hermitage Museum", "Russia", True),
    ("Lake Baikal", "Russia", False),
    ("Tivoli Gardens", "Denmark", False),
    ("Nyhavn", "Denmark", False),
    ("Little Mermaid statue", "Denmark", True),
    ("Vasa Museum", "Sweden", True),
    ("Stockholm City Hall", "Sweden", False),
    ("Preikestolen", "Norway", False),
    ("Vigeland Park", "Norway", False),
    ("Blue Lagoon", "Iceland", True),
    ("Hallgrimskirkja", "Iceland", False),
    ("Cliffs of Moher", "Ireland", True),
    ("Blarney Castle", "Ireland", False),
    ("Bran Castle", "Romania", False),
    ("Peles Castle", "Romania", False),
    ("Rila Monastery", "Bulgaria", False),
    ("Plitvice Lakes National Park", "Croatia", False),
    ("Diocletian's Palace", "Croatia", False),
    ("Lake Bled", "Slovenia", False),
    ("Stari Most", "Bosnia and Herzegovina", False),
    ("Monte Carlo Casino", "Monaco", True),
    # --- Asia ---
    ("Forbidden City", "China", True),
    ("Terracotta Army", "China", True),
    ("Temple of Heaven", "China", True),
    ("Potala Palace", "China", True),
    ("Summer Palace", "China", True),
    ("Leshan Giant Buddha", "China", True),
    ("Mount Fuji", "Japan", False),
    ("Fushimi Inari Shrine", "Japan", False),
    ("Himeji Castle", "Japan", False),
    ("Tokyo Skytree", "Japan", False),
    ("Itsukushima Shrine", "Japan", False),
    ("Nara Park", "Japan", False),
    ("Gyeongbokgung Palace", "South Korea", False),
    ("N Seoul Tower", "South Korea", False),
    ("Bulguksa Temple", "South Korea", False),
    ("Taj Mahal", "India", True),
    ("Red Fort", "India", True),
    ("Hawa Mahal", "India", False),
    ("Amber Fort", "India", False),
    ("Golden Temple", "India", True),
    ("Qutub Minar", "India", False),
    ("Boudhanath Stupa", "Nepal", True),
    ("Lumbini", "Nepal", False),
    ("Paro Taktsang", "Bhutan", False),
    ("Sigiriya", "Sri Lanka", False),
    ("Temple of the Tooth", "Sri Lanka", True),
    ("Badshahi Mosque", "Pakistan", True),
    ("Faisal Mosque", "Pakistan", True),
    ("Shwedagon Pagoda", "Myanmar", True),
    ("Bagan", "Myanmar", False),
    ("Wat Arun", "Thailand", False),
    ("Wat Pho", "Thailand", False),
    ("Angkor Wat", "Cambodia", False),
    ("Ha Long Bay", "Vietnam", False),
    ("Petronas Towers", "Malaysia", True),
    ("Batu Caves", "Malaysia", True),
    ("Marina Bay Sands", "Singapore", False),
    ("Gardens by the Bay", "Singapore", False),
    ("Merlion Park", "Singapore", False),
    ("Borobudur", "Indonesia", False),
    ("Komodo National Park", "Indonesia", False),
    ("Mount Bromo", "Indonesia", False),
    ("Chocolate Hills", "Philippines", True),
    ("Banaue Rice Terraces", "Philippines", True),
    ("Boracay", "Philippines", False),
    ("Genghis Khan Equestrian Statue", "Mongolia", True),
    ("Baiterek Tower", "Kazakhstan", True),
    ("Registan Square", "Uzbekistan", False),
    ("Persepolis", "Iran", False),
    ("Petra", "Jordan", False),
    ("Wadi Rum", "Jordan", False),
    ("Masada", "Israel", False),
    ("Baalbek", "Lebanon", False),
    ("Kaaba", "Saudi Arabia", True),
    ("Museum of Islamic Art", "Qatar", True),
    ("Burj Khalifa", "United Arab Emirates", True),
    ("Sheikh Zayed Grand Mosque", "United Arab Emirates", True),
    ("Sultan Qaboos Grand Mosque", "Oman", True),
    # --- Africa ---
    ("Pyramids of Giza", "Egypt", True),
    ("Great Sphinx of Giza", "Egypt", True),
    ("Valley of the Kings", "Egypt", True),
    ("Abu Simbel", "Egypt", False),
    ("Karnak Temple", "Egypt", False),
    ("Hassan II Mosque", "Morocco", True),
    ("Jemaa el-Fnaa", "Morocco", False),
    ("Chefchaouen", "Morocco", False),
    ("Ruins of Carthage", "Tunisia", True),
    ("Rock-Hewn Churches of Lalibela", "Ethiopia", True),
    ("Maasai Mara", "Kenya", True),
    ("Mount Kilimanjaro", "Tanzania", False),
    ("Serengeti National Park", "Tanzania", False),
    ("Sossusvlei", "Namibia", False),
    ("Okavango Delta", "Botswana", True),
    ("Table Mountain", "South Africa", False),
    ("Robben Island", "South Africa", False),
    ("Kruger National Park", "South Africa", False),
    ("Great Mosque of Djenne", "Mali", True),
    # --- Americas ---
    ("Statue of Liberty", "United States", True),
    ("Golden Gate Bridge", "United States", True),
    ("Grand Canyon", "United States", True),
    ("Yellowstone National Park", "United States", False),
    ("Mount Rushmore", "United States", False),
    ("Times Square", "United States", False),
    ("Hollywood Sign", "United States", True),
    ("White House", "United States", True),
    ("Antelope Canyon", "United States", False),
    ("Space Needle", "United States", True),
    ("CN Tower", "Canada", True),
    ("Banff National Park", "Canada", False),
    ("Chichen Itza", "Mexico", False),
    ("Teotihuacan", "Mexico", False),
    ("Frida Kahlo Museum", "Mexico", True),
    ("Tikal", "Guatemala", False),
    ("Great Blue Hole", "Belize", True),
    ("Arenal Volcano", "Costa Rica", False),
    ("Salt Cathedral of Zipaquira", "Colombia", True),
    ("Galapagos Islands", "Ecuador", True),
    ("Machu Picchu", "Peru", False),
    ("Nazca Lines", "Peru", True),
    ("Salar de Uyuni", "Bolivia", False),
    ("Easter Island", "Chile", False),
    ("Torres del Paine", "Chile", False),
    ("Atacama Desert", "Chile", True),
    ("Perito Moreno Glacier", "Argentina", True),
    ("Casa Rosada", "Argentina", False),
    ("Christ the Redeemer", "Brazil", False),
    ("Sugarloaf Mountain", "Brazil", False),
    ("Copacabana Beach", "Brazil", False),
    ("Angel Falls", "Venezuela", False),
    # --- Oceania ---
    ("Sydney Opera House", "Australia", True),
    ("Sydney Harbour Bridge", "Australia", True),
    ("Uluru", "Australia", False),
    ("Great Barrier Reef", "Australia", True),
    ("Bondi Beach", "Australia", False),
    ("Milford Sound", "New Zealand", False),
    ("Hobbiton Movie Set", "New Zealand", True),
]

# Synthetic neutral persons (no real celebrities; curated so that no name token
# collides with any landmark/country content token -- e.g. no "Jordan", "Chad",
# "Petra", "Amber", "Charles", "Frank", "Khan", "Park"; pinned by
# validate_pools).
PERSONS = [
    "Alex Chen", "Maria Novak", "Liam Okafor", "Elena Petrov", "Kenji Tanaka",
    "Aisha Haddad", "Diego Alvarez", "Hana Kowalski", "Felix Larsen",
    "Ingrid Weber", "Tomas Fischer", "Lena Moreau", "Ravi Patel",
    "Mei Nakamura", "Jonas Lindqvist", "Amara Osei", "Viktor Ivanov",
    "Leila Dubois", "Mateo Silva", "Freya Andersen", "Idris Mensah",
    "Carmen Rossi", "Yuki Yamamoto", "Nadia Rahim", "Oskar Novotny",
    "Talia Weiss", "Ruben Ortega", "Anouk Visser", "Selin Demir",
    "Bruno Ricci", "Alina Popescu", "Kofi Boateng", "Marta Horvat",
    "Emil Berg", "Rina Kapoor", "Stefan Muller", "Ayla Kaya",
    "Hugo Fontaine", "Nia Abara", "Casper Olsen", "Dalia Nassar",
    "Arjun Iyer", "Wren Holloway", "Iris Kalman", "Omar Karim",
    "Sana Qureshi", "Milo Vasquez", "Greta Lindholm", "Tariq Hamdan",
    "Vera Sokolov", "Dario Bianchi", "Noor Sabbagh", "Ewan Frazier",
    "Zofia Wojcik", "Rui Cardoso", "Esther Adeyemi", "Bela Farkas",
    "Yara Antoun", "Nils Ekberg", "Priya Raman",
]

# ---------------------------------------------------------------------------
# Countries distractor-plausibility rule (adversarial review 2026-08-04).
# ---------------------------------------------------------------------------
# Generic HEAD tokens of landmark names, in content_words()-normalised form
# (plural-stripped: "towers"->"tower", "falls"->"fall", "churches"->"church").
# Adjectives ("grand", "great", "golden") are NOT heads and stay out; every
# multi-word landmark is covered by its head noun instead.
GENERIC_HEAD_TOKENS = frozenset("""
tower bridge palace castle cathedral basilica museum temple mosque church
monastery shrine pagoda stupa fort square park garden gate statue fountain
casino gallery house building hall library tomb pyramid sphinx needle sign
arch wall city
island lake mount mountain beach bay canyon desert glacier valley fall reef
sound hill cave terrace delta lagoon fjord volcano hole ruin
""".split())

# One continent per table country (border-straddlers pinned once: Russia ->
# Europe, Turkey -> Asia; the assignment only shapes distractor plausibility).
CONTINENTS = {
    "France": "Europe", "United Kingdom": "Europe", "Italy": "Europe",
    "Spain": "Europe", "Portugal": "Europe", "Germany": "Europe",
    "Netherlands": "Europe", "Belgium": "Europe", "Switzerland": "Europe",
    "Austria": "Europe", "Czech Republic": "Europe", "Poland": "Europe",
    "Hungary": "Europe", "Greece": "Europe", "Turkey": "Asia",
    "Russia": "Europe", "Denmark": "Europe", "Sweden": "Europe",
    "Norway": "Europe", "Iceland": "Europe", "Ireland": "Europe",
    "Romania": "Europe", "Bulgaria": "Europe", "Croatia": "Europe",
    "Slovenia": "Europe", "Bosnia and Herzegovina": "Europe",
    "Monaco": "Europe",
    "China": "Asia", "Japan": "Asia", "South Korea": "Asia", "India": "Asia",
    "Nepal": "Asia", "Bhutan": "Asia", "Sri Lanka": "Asia",
    "Pakistan": "Asia", "Myanmar": "Asia", "Thailand": "Asia",
    "Cambodia": "Asia", "Vietnam": "Asia", "Malaysia": "Asia",
    "Singapore": "Asia", "Indonesia": "Asia", "Philippines": "Asia",
    "Mongolia": "Asia", "Kazakhstan": "Asia", "Uzbekistan": "Asia",
    "Iran": "Asia", "Jordan": "Asia", "Israel": "Asia", "Lebanon": "Asia",
    "Saudi Arabia": "Asia", "Qatar": "Asia",
    "United Arab Emirates": "Asia", "Oman": "Asia",
    "Egypt": "Africa", "Morocco": "Africa", "Tunisia": "Africa",
    "Ethiopia": "Africa", "Kenya": "Africa", "Tanzania": "Africa",
    "Namibia": "Africa", "Botswana": "Africa", "South Africa": "Africa",
    "Mali": "Africa",
    "United States": "Americas", "Canada": "Americas", "Mexico": "Americas",
    "Guatemala": "Americas", "Belize": "Americas", "Costa Rica": "Americas",
    "Colombia": "Americas", "Ecuador": "Americas", "Peru": "Americas",
    "Bolivia": "Americas", "Chile": "Americas", "Argentina": "Americas",
    "Brazil": "Americas", "Venezuela": "Americas",
    "Australia": "Oceania", "New Zealand": "Oceania",
}


def generic_tokens(landmark_name: str) -> set:
    """The landmark's generic head tokens (may be empty: Uluru, Petra ...)."""
    return content_words(landmark_name) & GENERIC_HEAD_TOKENS


def country_generic_tokens() -> dict:
    """country -> union of generic head tokens over its table landmarks."""
    out: dict = {}
    for name, country, _ in LANDMARKS:
        out.setdefault(country, set()).update(generic_tokens(name))
    return out


COUNTRY_GENERIC = country_generic_tokens()


def sample_distractors(rng: random.Random, gold_country: str,
                       gold_landmark: str, all_countries: list) -> tuple:
    """Tiered distractor sampling (module docstring, 'DISTRACTOR PLAUSIBILITY
    RULE'). Returns ([(country, source), ...] of length 3, sorted gold tokens).
    Sources: shared_token > same_continent > global (last resort, Oceania)."""
    toks = generic_tokens(gold_landmark)
    shared_pool = sorted(
        c for c in all_countries
        if c != gold_country and (COUNTRY_GENERIC.get(c, set()) & toks))
    continent_pool = sorted(
        c for c in all_countries
        if c != gold_country and CONTINENTS[c] == CONTINENTS[gold_country]
        and c not in shared_pool)
    chosen: list = []
    take = min(3, len(shared_pool))
    chosen += [(c, "shared_token") for c in rng.sample(shared_pool, take)]
    if len(chosen) < 3:
        take = min(3 - len(chosen), len(continent_pool))
        chosen += [(c, "same_continent")
                   for c in rng.sample(continent_pool, take)]
    if len(chosen) < 3:
        got = {c for c, _ in chosen}
        rest_pool = sorted(c for c in all_countries
                           if c != gold_country and c not in got)
        chosen += [(c, "global")
                   for c in rng.sample(rest_pool, 3 - len(chosen))]
    return chosen, sorted(toks)


# ---------------------------------------------------------------------------
# Tip Sheets task.
# ---------------------------------------------------------------------------
# 80 synthetic companies = 40 invented roots x 2 suffixes each (deterministic
# pairing below). Roots invented; suffixes screened against SIGNAL_WORDS and
# choice tokens (NOTE: no "Holdings" -- its token stems to "holding", a signal
# word; no "Partners" -- too close to "partnership").
_COMPANY_ROOTS = [
    "Norvex", "Quellar", "Tarvos", "Veltrix", "Kaldera", "Mirovex", "Sunthra",
    "Pellax", "Dornick", "Halcyor", "Vexley", "Ostrix", "Quanterra", "Nimbex",
    "Corvane", "Zephrix", "Altrion", "Fentara", "Gryphex", "Krellon",
    "Lumetra", "Morvath", "Nexavo", "Orvatek", "Prellium", "Quorvex",
    "Ravlin", "Sorvello", "Trevanta", "Ulvex", "Vandrel", "Werlano",
    "Xantrell", "Yorvane", "Zalvor", "Ambrix", "Bellavox", "Cindral",
    "Drovana", "Elstrom",
]
_COMPANY_SUFFIXES = [
    "Systems", "Dynamics", "Industries", "Laboratories", "Technologies",
    "Analytics", "Materials", "Logistics", "Robotics", "Networks",
    "Aerospace", "Biotech", "Group", "Ventures", "Instruments", "Components",
    "Manufacturing", "Microsystems", "Fabrication", "Optics",
]
COMPANIES = [f"{root} {_COMPANY_SUFFIXES[(2 * i + j) % len(_COMPANY_SUFFIXES)]}"
             for i, root in enumerate(_COMPANY_ROOTS) for j in (0, 1)]

EVENTS_POSITIVE = ("new product launch", "major partnership")
EVENTS_NEGATIVE = ("lawsuit", "CEO resigned", "product recall")
EVENTS = EVENTS_POSITIVE + EVENTS_NEGATIVE

# DEVIATION FROM SPEC (flagged): the neutral analyst rating is "unchanged",
# not "hold". A rating literally spelled "hold" would put the HOLD choice
# verbatim inside every third evidence line, violating the no-choice-string-
# in-evidence leakage assert (a string-matching receiver could answer HOLD
# without applying the rule). The rule only ever tests "downgrade", so the
# semantics are identical.
RATINGS = ("upgrade", "downgrade", "unchanged")
REVENUE_PCTS = tuple(range(2, 25))          # 2..24 percent
TIP_LABELS = ("BUY", "SELL", "HOLD")
TIP_DISTRACTOR = "NO DATA"                  # 4th choice, never gold

# The deterministic profile -> label rule (BENCH task-2 spec, verbatim), stated
# inside every sheet so the gold is derivable from the private sheet ALONE.
# Phrased as "buying/selling/holding" so no choice string appears verbatim.
RULE_TEXT = (
    "Decision rule: buying is advised when revenue is up and the event is "
    "positive (new product launch or major partnership) and the analyst "
    "rating is not a downgrade; selling is advised when revenue is down and "
    "the event is negative (lawsuit, CEO resigned, or product recall); "
    "otherwise holding is advised.")

# Signal vocabulary that must NEVER appear in a receiver-visible question
# (word-level, content_words-normalised).
SIGNAL_WORDS = frozenset("""
revenue up down lawsuit launch product partnership recall resigned
resignation ceo analyst rating upgrade downgrade unchanged buying selling
holding positive negative advised decision rule event percent
""".split())

# Framing tokens the tip-sheet question legitimately shares with the sheet
# header ("According to the tip sheet ..." / "TIP SHEET (private)").
TIP_FRAMING_TOKENS = frozenset({"tip", "sheet"})


def recommend(direction: str, event: str, rating: str) -> str:
    """The deterministic tip-sheet rule. BUY iff revenue up AND positive event
    AND not a downgrade; SELL iff revenue down AND negative event; else HOLD.
    (Revenue pct magnitude is decorative by spec: only the direction enters.)
    """
    if direction not in ("up", "down"):
        raise ValueError(f"bad direction {direction!r}")
    if event not in EVENTS:
        raise ValueError(f"bad event {event!r}")
    if rating not in RATINGS:
        raise ValueError(f"bad rating {rating!r}")
    if direction == "up" and event in EVENTS_POSITIVE and rating != "downgrade":
        return "BUY"
    if direction == "down" and event in EVENTS_NEGATIVE:
        return "SELL"
    return "HOLD"


# ------------------------------------------------------------ pool validation

def _tokens(strings) -> set:
    out = set()
    for s in strings:
        out |= content_words(s)
    return out


def validate_pools() -> dict:
    """Mechanical checks on the hand-written tables; SystemExit on violation.
    Returns pool sizes for the summary json."""
    names = [n for n, _, _ in LANDMARKS]
    countries = sorted({c for _, c, _ in LANDMARKS})
    problems = []
    if len(LANDMARKS) < 120:
        problems.append(f"only {len(LANDMARKS)} landmarks (<120)")
    if len(set(n.lower() for n in names)) != len(names):
        problems.append("duplicate landmark names")
    if len(set(PERSONS)) != len(PERSONS):
        problems.append("duplicate person names")
    if len(set(COMPANIES)) != len(COMPANIES):
        problems.append("duplicate company names")
    # Leakage rule: no landmark contains ANY table country's name (raw
    # lowercase substring -- the "Great Wall of China" rule, generalised to
    # every country so a wrong-country string can never ride in the evidence).
    for lm in names:
        for c in countries:
            if c.lower() in lm.lower():
                problems.append(f"landmark {lm!r} contains country {c!r}")
    # Token-level: choice-side tokens (countries) must never occur in
    # evidence-side tokens (persons + landmarks).
    country_toks = _tokens(countries)
    lm_toks = _tokens(names)
    person_toks = _tokens(PERSONS)
    for bad in sorted(lm_toks & country_toks):
        problems.append(f"landmark token {bad!r} collides with a country token")
    for bad in sorted(person_toks & (country_toks | lm_toks)):
        problems.append(f"person token {bad!r} collides with landmark/country")
    # Distractor rule inputs: every table country has exactly one continent,
    # and generic head tokens never collide with a country token.
    if set(CONTINENTS) != set(countries):
        problems.append(
            f"CONTINENTS keys != table countries (missing "
            f"{sorted(set(countries) - set(CONTINENTS))}, extra "
            f"{sorted(set(CONTINENTS) - set(countries))})")
    bad_cont = sorted(set(CONTINENTS.values()) -
                      {"Europe", "Asia", "Africa", "Americas", "Oceania"})
    if bad_cont:
        problems.append(f"unknown continents {bad_cont}")
    for bad in sorted(GENERIC_HEAD_TOKENS & country_toks):
        problems.append(f"generic head token {bad!r} is also a country token")
    # Tip sheets: company tokens must carry no signal word and no choice token.
    choice_toks = _tokens(TIP_LABELS + (TIP_DISTRACTOR,))
    for comp in COMPANIES:
        bad = content_words(comp) & (SIGNAL_WORDS | choice_toks)
        if bad:
            problems.append(f"company {comp!r} carries signal/choice tokens {sorted(bad)}")
    # No choice string may hide inside the frozen rule text.
    for lab in TIP_LABELS + (TIP_DISTRACTOR,):
        if contains_verbatim(norm_text(RULE_TEXT), lab):
            problems.append(f"choice {lab!r} appears verbatim in RULE_TEXT")
    if "hold" in RATINGS:
        problems.append("rating 'hold' collides with choice HOLD (use 'unchanged')")
    if problems:
        raise SystemExit("ABORT: pool validation failed:\n  " +
                         "\n  ".join(problems))
    return {"n_landmarks": len(LANDMARKS), "n_countries": len(countries),
            "n_persons": len(PERSONS), "n_companies": len(COMPANIES)}


# --------------------------------------------------------------- RNG plumbing

def task_rng(task: str, seed: int) -> random.Random:
    """Stable sub-RNG per task (independent of the other task's draw count)."""
    h = hashlib.sha256(f"{seed}:{task}".encode("utf-8")).hexdigest()
    return random.Random(int(h[:16], 16))


def balanced_slots(n: int, rng: random.Random) -> list:
    """Exactly-balanced gold positions (n/4 per letter at n%4==0), shuffled."""
    slots = [i % N_CHOICES for i in range(n)]
    rng.shuffle(slots)
    return slots


def balanced_labels(n: int, rng: random.Random) -> list:
    """Exactly-balanced BUY/SELL/HOLD targets (each >= 25% by construction)."""
    labels = [TIP_LABELS[i % len(TIP_LABELS)] for i in range(n)]
    rng.shuffle(labels)
    return labels


# ------------------------------------------------------------- AC screens

def ac_flags_countries(person, landmark, question, choices, evidence) -> dict:
    q_toks, e_toks = content_words(question), content_words(evidence)
    choice_toks = _tokens(c["text"] for c in choices)
    return {
        "landmark_in_question": landmark.lower() in question.lower(),
        "extra_evidence_question_tokens":
            sorted((q_toks & e_toks) - content_words(person)),
        "choice_tokens_in_evidence": sorted(choice_toks & e_toks),
        "signal_words_in_question": [],          # countries: vacuous, uniform key
    }


def ac_flags_tipsheets(company, question, choices, evidence) -> dict:
    q_toks, e_toks = content_words(question), content_words(evidence)
    choice_toks = _tokens(c["text"] for c in choices)
    return {
        "landmark_in_question": False,           # tipsheets: vacuous, uniform key
        "extra_evidence_question_tokens":
            sorted((q_toks & e_toks) - content_words(company)
                   - TIP_FRAMING_TOKENS),
        "choice_tokens_in_evidence": sorted(choice_toks & e_toks),
        "signal_words_in_question": sorted(q_toks & SIGNAL_WORDS),
    }


def make_episode(benchmark, split, idx, evidence, question, choice_texts,
                 gold_slot, ac_flags, ac_meta) -> dict:
    choices = [{"label": l, "text": t} for l, t in zip(LABELS, choice_texts)]
    answer_key = LABELS[gold_slot]
    flags = screen_leakage(evidence, choices, answer_key, question)
    flags.update(ac_flags)
    flags["screen"] = EVIDENCE_SOURCE
    flags["violation"] = bool(
        flags["violation_strict"] or flags["landmark_in_question"]
        or flags["extra_evidence_question_tokens"]
        or flags["choice_tokens_in_evidence"]
        or flags["signal_words_in_question"])
    return {
        "item_id": f"{benchmark}/{split}/{idx:06d}",
        "benchmark": benchmark,
        "split": split,
        "sender_evidence": evidence,
        "evidence_source": EVIDENCE_SOURCE,
        "question_stem": question,
        "choices": choices,
        "answerKey": answer_key,
        "leakage_flags": flags,
        "item_sha8": item_sha8(question, choices, answer_key),
        "ac_meta": ac_meta,
    }


# ------------------------------------------------------------- the builders

def build_countries(n_dev: int = 400, n_test: int = 400,
                    seed: int = SEED) -> dict:
    """-> {"dev": [episodes], "test": [episodes]}, disjoint (person,landmark)
    combos across splits (sampled without replacement from the cross product)."""
    rng = task_rng("countries", seed)
    pairs = [(p, lm) for p in PERSONS for lm in LANDMARKS]
    if n_dev + n_test > len(pairs):
        raise SystemExit(f"ABORT: requested {n_dev + n_test} episodes > "
                         f"{len(pairs)} distinct (person,landmark) combos")
    rng.shuffle(pairs)
    picked = {"dev": pairs[:n_dev], "test": pairs[n_dev:n_dev + n_test]}
    all_countries = sorted({c for _, c, _ in LANDMARKS})
    out = {}
    for split, split_pairs in picked.items():
        slots = balanced_slots(len(split_pairs), rng)
        eps = []
        for i, ((person, (lm_name, country, use_the)), slot) in enumerate(
                zip(split_pairs, slots)):
            evidence = (f"{person} is at "
                        f"{'the ' if use_the else ''}{lm_name}.")
            question = f"Which country is {person} in?"
            distractors, gold_toks = sample_distractors(
                rng, country, lm_name, all_countries)
            texts = [None] * N_CHOICES
            texts[slot] = country
            rest = [c for c, _ in distractors]
            rng.shuffle(rest)
            for j in range(N_CHOICES):
                if texts[j] is None:
                    texts[j] = rest.pop()
            eps.append(make_episode(
                "ac-countries", split, i, evidence, question, texts, slot,
                ac_flags_countries(person, lm_name, question,
                                   [{"label": l, "text": t}
                                    for l, t in zip(LABELS, texts)], evidence),
                {"person": person, "landmark": lm_name, "country": country,
                 "use_the": use_the,
                 "gold_generic_tokens": gold_toks,
                 "distractors": [{"country": c, "source": s}
                                 for c, s in distractors]}))
        out[split] = eps
    return out


def _sample_profile(rng: random.Random) -> tuple:
    return (rng.choice(("up", "down")), rng.choice(REVENUE_PCTS),
            rng.choice(EVENTS), rng.choice(RATINGS))


def _sample_profile_with_label(rng: random.Random, label: str) -> tuple:
    """Rejection sampling until the deterministic rule yields `label`
    (BUY p~0.13, SELL p~0.30, HOLD p~0.57 per draw -- converges fast)."""
    while True:
        prof = _sample_profile(rng)
        if recommend(prof[0], prof[2], prof[3]) == label:
            return prof


def _sheet_text(companies: list) -> str:
    lines = [f"TIP SHEET (private). {RULE_TEXT}"]
    for c in companies:
        lines.append(f"- {c['name']}: revenue {c['revenue_direction']} "
                     f"{c['revenue_pct']}%; event: {c['event']}; "
                     f"analyst rating: {c['rating']}.")
    return "\n".join(lines)


def build_tipsheets(n_dev: int = 400, n_test: int = 400,
                    seed: int = SEED) -> dict:
    """-> {"dev": [...], "test": [...]}. The asked-company label is scheduled
    exactly balanced over BUY/SELL/HOLD (>=25% each split); NO DATA pads the
    choice set to 4 and is never gold. All three (company, profile) combos of
    every sheet are globally unseen -> splits disjoint by construction."""
    rng = task_rng("tipsheets", seed)
    seen: set = set()
    out = {}
    for split, n in (("dev", n_dev), ("test", n_test)):
        targets = balanced_labels(n, rng)
        slots = balanced_slots(n, rng)
        eps = []
        for i, (target, slot) in enumerate(zip(targets, slots)):
            while True:
                names = rng.sample(COMPANIES, 3)
                asked_idx = rng.randrange(3)
                profiles = [(_sample_profile_with_label(rng, target)
                             if j == asked_idx else _sample_profile(rng))
                            for j in range(3)]
                keys = {(names[j], profiles[j]) for j in range(3)}
                if len(keys) == 3 and not (keys & seen):
                    seen |= keys
                    break
            companies = []
            for j in range(3):
                d, pct, ev, rat = profiles[j]
                companies.append({
                    "name": names[j], "revenue_direction": d,
                    "revenue_pct": pct, "event": ev, "rating": rat,
                    "label": recommend(d, ev, rat)})
            asked = companies[asked_idx]
            assert asked["label"] == target        # schedule honoured
            evidence = _sheet_text(companies)
            question = (f"According to the tip sheet, what is the recommended "
                        f"action for {asked['name']}?")
            others = [l for l in TIP_LABELS if l != target] + [TIP_DISTRACTOR]
            rng.shuffle(others)
            texts = [None] * N_CHOICES
            texts[slot] = target
            for j in range(N_CHOICES):
                if texts[j] is None:
                    texts[j] = others.pop()
            eps.append(make_episode(
                "ac-tipsheets", split, i, evidence, question, texts, slot,
                ac_flags_tipsheets(asked["name"], question,
                                   [{"label": l, "text": t}
                                    for l, t in zip(LABELS, texts)], evidence),
                {"companies": companies, "asked": asked["name"],
                 "asked_label": target}))
        out[split] = eps
    return out


# ---------------------------------------------------------- stats + writing

def ac_screen_summary(eps: list) -> dict:
    return {
        "n_violation": sum(e["leakage_flags"]["violation"] for e in eps),
        "n_landmark_in_question": sum(
            e["leakage_flags"]["landmark_in_question"] for e in eps),
        "n_extra_overlap_tokens": sum(
            bool(e["leakage_flags"]["extra_evidence_question_tokens"])
            for e in eps),
        "n_choice_token_in_evidence": sum(
            bool(e["leakage_flags"]["choice_tokens_in_evidence"])
            for e in eps),
        "n_signal_words_in_question": sum(
            bool(e["leakage_flags"]["signal_words_in_question"])
            for e in eps),
    }


def assert_zero_violations(benchmark: str, split: str, eps: list) -> None:
    bad = [e["item_id"] for e in eps if e["leakage_flags"]["violation"]]
    if bad:
        raise SystemExit(
            f"ABORT: {benchmark}/{split}: {len(bad)} leakage violations in a "
            f"MECHANICAL generator (must be exactly 0; first: {bad[:5]}). "
            "The word lists or templates leak -- fix them, do not ship.")


def split_stats(benchmark: str, split: str, eps: list, pool_sizes: dict,
                seed: int) -> dict:
    stats = {
        "benchmark": benchmark, "split": split, "task": benchmark,
        "seed": seed, "n_built": len(eps),
        "evidence_source": EVIDENCE_SOURCE,
        "published_n": PUBLISHED[benchmark]["published_n"],
        "protocol": PUBLISHED[benchmark]["protocol"],
        "pool_sizes": pool_sizes,
        "answer_position": assert_choice_balance(eps),
        "leakage": leakage_summary(eps),
        "ac_screens": ac_screen_summary(eps),
        "one_touch_test_split": split == "test",
        "plan": PLAN,
    }
    if benchmark == "ac-tipsheets":
        dist = {l: 0 for l in TIP_LABELS}
        for e in eps:
            dist[e["ac_meta"]["asked_label"]] += 1
        stats["label_distribution"] = dist
        stats["min_label_frac"] = round(min(dist.values()) / max(len(eps), 1), 4)
    else:
        dist: dict = {}
        for e in eps:
            c = e["ac_meta"]["country"]
            dist[c] = dist.get(c, 0) + 1
        stats["gold_country_distribution"] = dict(sorted(dist.items()))
        n = max(len(eps), 1)

        def frac_with(source, quantifier):
            k = sum(quantifier(d["source"] == source
                               for d in e["ac_meta"]["distractors"])
                    for e in eps)
            return k, round(k / n, 4)

        ge1_shared = frac_with("shared_token", any)
        all_shared = frac_with("shared_token", all)
        cont = frac_with("same_continent", any)
        glob = frac_with("global", any)
        stats["distractor_rule"] = {
            "rule": "shared-generic-head-token countries first; "
                    "same-continent fallback; global last resort "
                    "(adversarial review 2026-08-04: generic-token answering)",
            "n_items_ge1_shared_token": ge1_shared[0],
            "frac_items_ge1_shared_token": ge1_shared[1],
            "n_items_all3_shared_token": all_shared[0],
            "frac_items_all3_shared_token": all_shared[1],
            "n_items_continent_fallback": cont[0],
            "frac_items_continent_fallback": cont[1],
            "n_items_global_fallback": glob[0],
            "frac_items_global_fallback": glob[1],
        }
    return stats


def write_split(out_dir: str, benchmark: str, split: str, eps: list,
                stats: dict, argv: list) -> tuple:
    out = os.path.join(out_dir, f"{benchmark}_{split}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    write_jsonl(out, eps)
    stats.update({
        "out": os.path.abspath(out), "created_utc": utcnow(),
        "jsonl_sha256": sha256_file(out),
        "builder_sha256": sha256_file(os.path.abspath(__file__)),
        "argv": list(argv),
    })
    sum_path = out.replace(".jsonl", "") + ".summary.json"
    with open(sum_path, "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return out, sum_path


# --------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Mechanically generate the AC-suite episode JSONLs "
                    "(BENCH_PLAN §1.1 co-primary; published protocol rebuilt).")
    ap.add_argument("--tasks", nargs="+", default=["countries", "tipsheets"],
                    choices=["countries", "tipsheets"])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"],
                    choices=["dev", "test"])
    ap.add_argument("--n-dev", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", default="data/bench")
    ap.add_argument("--allow-test", action="store_true",
                    help="acknowledge that the test split is the one-touch "
                         "confirmatory set (Hard Rule 6): files are written "
                         "but nothing may be run on them before the "
                         "confirmatory launch")
    return ap


BUILDERS = {"countries": ("ac-countries", build_countries),
            "tipsheets": ("ac-tipsheets", build_tipsheets)}


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if "test" in a.splits and not a.allow_test:
        raise SystemExit(
            "ABORT: split 'test' is the one-touch confirmatory set (Hard "
            "Rule 6; BENCH_PLAN §1.2). Pass --allow-test to WRITE the file; "
            "running anything on it stays forbidden until the confirmatory "
            "launch.")
    pool_sizes = validate_pools()
    argv_rec = sys.argv[1:] if argv is None else list(argv)
    for task in a.tasks:
        benchmark, builder = BUILDERS[task]
        # dev and test are always generated JOINTLY (disjointness needs the
        # shared draw); --splits only controls which files are written.
        splits = builder(n_dev=a.n_dev, n_test=a.n_test, seed=a.seed)
        for split in a.splits:
            eps = splits[split]
            assert_zero_violations(benchmark, split, eps)
            stats = split_stats(benchmark, split, eps, pool_sizes, a.seed)
            out, sum_path = write_split(a.out_dir, benchmark, split, eps,
                                        stats, argv_rec)
            lk, ac = stats["leakage"], stats["ac_screens"]
            print(f"[ac] {benchmark}/{split}: rows={len(eps)} "
                  f"letters={stats['answer_position']['fractions']} "
                  + (f"labels={stats['label_distribution']} "
                     if task == "tipsheets" else
                     f"countries={len(stats['gold_country_distribution'])} "
                     f"distractors[ge1_shared="
                     f"{stats['distractor_rule']['frac_items_ge1_shared_token']}"
                     f" all3_shared="
                     f"{stats['distractor_rule']['frac_items_all3_shared_token']}"
                     f" cont_fb="
                     f"{stats['distractor_rule']['frac_items_continent_fallback']}"
                     f" glob_fb="
                     f"{stats['distractor_rule']['frac_items_global_fallback']}] ")
                  + f"-> {out}", flush=True)
            print(f"[ac] leakage (§1.2 + AC screens): "
                  f"violations={ac['n_violation']} "
                  f"gold_verbatim={lk['n_gold_verbatim']} "
                  f"distractor_verbatim={lk['n_distractor_verbatim']} "
                  f"answer_letter={lk['n_answer_letter_pattern']} "
                  f"landmark_in_q={ac['n_landmark_in_question']} "
                  f"signal_in_q={ac['n_signal_words_in_question']} "
                  f"extra_overlap={ac['n_extra_overlap_tokens']} "
                  f"choice_tok_in_ev={ac['n_choice_token_in_evidence']} "
                  f"overlap_ratio p50="
                  f"{lk['overlap_ratio_quantiles']['p50']} "
                  f"p90={lk['overlap_ratio_quantiles']['p90']}", flush=True)
            print(f"[ac] summary -> {sum_path}", flush=True)
    print("[ac] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
