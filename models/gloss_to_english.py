"""Turn recognised sign glosses into a readable English sentence (rule-based, offline).

ASL word order and grammar differ from English: there is no "is", no articles, and time words often
come first. This converter adds the copula, verb agreement and tense, articles and prepositions for
the common patterns in our vocabulary, e.g.

    minemy brother sick          -> My brother is sick.
    dad go store tomorrow        -> Dad will go to the store tomorrow.
    yesterday weus ride car home -> We rode the car home yesterday.

It is an approximation for display; recognition and WER are always measured on the glosses.
"""

PEOPLE_NO_ARTICLE = {'mom', 'dad', 'grandma', 'grandpa', 'brother', 'aunt', 'uncle'}
PEOPLE = PEOPLE_NO_ARTICLE | {'girl', 'boy', 'child', 'man', 'person', 'police', 'fireman', 'cowboy',
                              'clown', 'kitty', 'puppy', 'dog', 'cat'}
PRONOUNS = {'minemy': 'I', 'weus': 'we', 'hesheit': 'they', 'yourself': 'you'}
ADJECTIVES = {'happy', 'sad', 'sick', 'hungry', 'thirsty', 'sleepy', 'mad', 'hot', 'cute', 'dirty', 'clean',
              'wet', 'dry', 'fine', 'loud', 'quiet', 'old', 'pretty', 'sticky', 'empty', 'noisy', 'awake',
              'bad', 'better', 'fast', 'high', 'yucky', 'stuck', 'same', 'red', 'blue', 'green', 'yellow',
              'white', 'black', 'brown', 'orange'}
# base: (3rd person singular, past)
VERBS = {
    'go': ('goes', 'went'), 'drink': ('drinks', 'drank'), 'sleep': ('sleeps', 'slept'), 'read': ('reads', 'read'),
    'like': ('likes', 'liked'), 'have': ('has', 'had'), 'see': ('sees', 'saw'), 'give': ('gives', 'gave'),
    'find': ('finds', 'found'), 'wait': ('waits', 'waited'), 'ride': ('rides', 'rode'), 'cry': ('cries', 'cried'),
    'jump': ('jumps', 'jumped'), 'dance': ('dances', 'danced'), 'listen': ('listens', 'listened'),
    'look': ('looks', 'looked'), 'talk': ('talks', 'talked'), 'hide': ('hides', 'hid'), 'kiss': ('kisses', 'kissed'),
    'make': ('makes', 'made'), 'say': ('says', 'said'), 'think': ('thinks', 'thought'),
    'touch': ('touches', 'touched'), 'taste': ('tastes', 'tasted'), 'smile': ('smiles', 'smiled'),
    'hate': ('hates', 'hated'), 'open': ('opens', 'opened'), 'close': ('closes', 'closed'), 'cut': ('cuts', 'cut'),
    'fall': ('falls', 'fell'), 'drop': ('drops', 'dropped'), 'blow': ('blows', 'blew'), 'hear': ('hears', 'heard'),
    'wake': ('wakes up', 'woke up'), 'stay': ('stays', 'stayed'), 'pretend': ('pretends', 'pretended'),
    'nap': ('naps', 'napped'), 'finish': ('finishes', 'finished'), 'callonphone': ('calls', 'called'),
    'haveto': ('has to', 'had to'),
}
VERB_TEXT = {'callonphone': 'call', 'haveto': 'have to', 'wake': 'wake up'}
MASS_NOUNS = {'water', 'milk', 'food', 'pizza', 'icecream', 'cereal', 'gum', 'chocolate', 'frenchfries', 'nuts',
              'grass', 'snow', 'rain', 'jeans', 'pajamas'}
PLACES = {'store': 'to the store', 'bed': 'to bed', 'home': 'home', 'outside': 'outside', 'backyard': 'to the backyard',
          'bedroom': 'to the bedroom', 'room': 'to the room', 'farm': 'to the farm', 'pool': 'to the pool'}
NOUN_TEXT = {'icecream': 'ice cream', 'frenchfries': 'french fries', 'glasswindow': 'window', 'TV': 'TV',
             'callonphone': 'phone'}
TIMES = {'now': 'now', 'later': 'later', 'tomorrow': 'tomorrow', 'yesterday': 'yesterday', 'night': 'at night',
         'morning': 'in the morning'}
FUTURE, PAST = {'tomorrow', 'later'}, {'yesterday'}
INTERJECTIONS = {'hello': 'hello', 'bye': 'bye', 'please': 'please', 'thankyou': 'thank you', 'yes': 'yes',
                 'no': 'no', 'shhh': 'shh'}


def _noun(word, article):
    text = NOUN_TEXT.get(word, word)
    if word in MASS_NOUNS or article is None:
        return text
    if article == 'a' and text[0] in 'aeiou':
        return 'an ' + text
    return f'{article} {text}'


def _subject(tokens):
    """Read a subject from the front of tokens -> (text, person, used)."""
    if not tokens:
        return None, None, 0
    w = tokens[0]
    if w == 'minemy' and len(tokens) > 1 and tokens[1] not in VERBS and tokens[1] not in ADJECTIVES \
            and tokens[1] not in TIMES and tokens[1] not in INTERJECTIONS:
        return 'my ' + NOUN_TEXT.get(tokens[1], tokens[1]), 3, 2
    if w in PRONOUNS:
        p = PRONOUNS[w]
        return p, (1 if p == 'I' else 2 if p in ('we', 'you', 'they') else 3), 1
    if w in PEOPLE:
        return (w if w in PEOPLE_NO_ARTICLE else 'the ' + w), 3, 1
    return None, None, 0


def _be(person, tense):
    if tense == 'past':
        return 'was' if person in (1, 3) else 'were'
    if tense == 'future':
        return 'will be'
    return {1: 'am', 2: 'are'}.get(person, 'is')


def _verb(word, person, tense):
    third, past = VERBS[word]
    if tense == 'past':
        return past
    if tense == 'future':
        return 'will ' + VERB_TEXT.get(word, word)
    return third if person == 3 else VERB_TEXT.get(word, word)


def _clause(tokens, tense, time_text):
    subject, person, used = _subject(tokens)
    rest = tokens[used:]
    vocative = None
    if subject is None and rest and rest[0] in VERBS:   # imperative, e.g. "wait please mom"
        person, tense = 2, 'present'
    predicates, i = [], 0
    while i < len(rest):
        w = rest[i]
        if w in ADJECTIVES:
            adj = w
            if w in ('hungry', 'thirsty') and i + 1 < len(rest) and rest[i + 1] not in VERBS and rest[i + 1] not in ADJECTIVES:
                predicates.append(f'{_be(person, tense) if subject else ""} {adj} for {_noun(rest[i + 1], None)}'.strip())
                i += 2
                continue
            predicates.append(f'{_be(person, tense)} {adj}' if subject else adj)
            i += 1
        elif w in VERBS:
            verb = _verb(w, person, tense) if subject else VERB_TEXT.get(w, w)
            objs, i = [], i + 1
            while i < len(rest) and rest[i] not in VERBS and rest[i] not in ADJECTIVES:
                objs.append(rest[i])
                i += 1
            if w == 'go' and objs and objs[0] in PLACES:
                phrase = PLACES[objs[0]] + (' ' + ' '.join(NOUN_TEXT.get(o, o) for o in objs[1:]) if objs[1:] else '')
            elif w == 'ride' and objs:
                tail = [PLACES.get(o, o) if o in ('home', 'outside') else o for o in objs[1:]]
                phrase = _noun(objs[0], 'the') + (' ' + ' '.join(tail) if tail else '')
            elif objs and objs[-1] in PEOPLE and subject is None:   # "wait please mom": mom is addressed
                vocative = objs.pop()
                phrase = ' '.join(_noun(o, None) for o in objs)
            elif objs:
                compound = ' '.join(NOUN_TEXT.get(o, o) for o in objs)
                first = objs[0]
                if first in PEOPLE:
                    phrase = (first if first in PEOPLE_NO_ARTICLE else 'the ' + first) + \
                        (' ' + ' '.join(objs[1:]) if objs[1:] else '')
                elif first in MASS_NOUNS:
                    phrase = compound
                else:
                    phrase = ('an ' if compound[0] in 'aeiou' else 'a ') + compound
            else:
                phrase = ''
            predicates.append(f'{verb} {phrase}'.strip())
        else:   # a noun with no verb before it
            predicates.append(_noun(w, None if subject is None else 'a'))
            i += 1
    body = ' and '.join(p for p in predicates if p)
    if subject:
        text = f'{subject} {body}'.strip()
    else:
        text = body
    if time_text:
        text = f'{text} {time_text}'.strip()
    return text, vocative


def to_english(glosses):
    """List of sign glosses (or a space-separated string) -> one English sentence."""
    if isinstance(glosses, str):
        glosses = glosses.split()
    tokens = [g for g in glosses if g]
    if not tokens:
        return ''
    times = [t for t in tokens if t in TIMES]
    tense = 'past' if any(t in PAST for t in times) else 'future' if any(t in FUTURE for t in times) else 'present'
    time_text = ' '.join(TIMES[t] for t in times)
    tokens = [t for t in tokens if t not in TIMES]

    # Interjections split the sentence into parts: "hello grandma thankyou" -> "Hello grandma, thank you."
    parts, current = [], []
    for t in tokens:
        if t in INTERJECTIONS:
            if current:
                parts.append(('clause', current))
                current = []
            parts.append(('interjection', INTERJECTIONS[t]))
        else:
            current.append(t)
    if current:
        parts.append(('clause', current))

    pieces, time_used = [], False
    for kind, value in parts:
        if kind == 'interjection':
            pieces.append(value)
            continue
        if len(value) == 1 and value[0] in PEOPLE and pieces and pieces[-1] in ('hello', 'bye', 'thank you'):
            pieces[-1] = f'{pieces[-1]} {value[0]}'   # "hello grandma"
            continue
        text, vocative = _clause(value, tense, '' if time_used else time_text)
        time_used = time_used or bool(time_text)
        pieces.append(text)
        if vocative:
            pieces.append(vocative)
    if time_text and not time_used:
        pieces.append(time_text)
    sentence = ', '.join(p for p in pieces if p)
    sentence = sentence[0].upper() + sentence[1:]
    return sentence + '.'
