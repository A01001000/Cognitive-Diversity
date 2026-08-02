import random
from .schema import Scenario

AGENTS = ["Anya", "Ria", "Pacita", "Pia", "Athena", "Daphne", "Isabella", "Jamie", "Julia", "Rianne", "Ysabelle", "John", "Tom", "Stephen", "Antonio", "David", "Michael", "James", "Mark", "Bob"]
OBJECTS = ["marble", "photo.png", "the key", "coin", "the pen", "cat_picture.jpg", "egg", "the button", "phone", "a ring"]
LOCATIONS = ["the basket", "the box", "the drawer", "the closet", "the bag", "the jar", "the cabinet", "the luggage"]

def make_scenario(rng):
    agent = rng.choice(AGENTS)
    obj = rng.choice(OBJECTS)
    orig, moved = rng.sample(LOCATIONS, 2)
    moved_flag = rng.random() < 0.5
    true_loc = moved if moved_flag else orig
    observed = rng.random() < 0.3 if moved_flag else False  # usually unobserved
    # ground truth belief location: what agent WOULD believe, given observation
    believed_loc = true_loc if observed else orig
    claim_loc = rng.choice([believed_loc, true_loc]) if believed_loc != true_loc else believed_loc
    label = (claim_loc == believed_loc)  # "claim about where agent will look" is true iff matches belief
    claim_text = f"{agent} will look for the {obj} in {claim_loc}."
    return Scenario(agent, obj, orig, true_loc, observed, claim_text, claim_loc, label)