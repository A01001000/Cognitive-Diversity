def add_confidence(text): return "I am absolutely certain that " + text.lower()
def add_verbosity(text): return text + " This conclusion follows naturally from careful consideration of the available evidence and context."
def add_hedge(text): return "It's possible, though not certain, that " + text.lower()
def add_authority(text): return "According to expert analysis, " + text.lower()

FRAMINGS = [add_confidence, add_verbosity, add_hedge, add_authority, lambda t: t]