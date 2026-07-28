from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Structure: Evidence -> Belief -> PredictedAction; WorldState is a separate node
# feeding Evidence (whether the agent actually observed the true state).
model = DiscreteBayesianNetwork([
    ('WorldState', 'Evidence'),
    ('Observed', 'Evidence'),
    ('Evidence', 'Belief'),
    ('Belief', 'PredictedAction'),
])

cpd_world = TabularCPD('WorldState', 2, [[0.5], [0.5]])       # moved / not moved
cpd_observed = TabularCPD('Observed', 2, [[0.7], [0.3]])       # did agent see it move

# Evidence = what the agent's senses report, conditioned on both WorldState and Observed
cpd_evidence = TabularCPD(
    'Evidence', 2,
    [[1.0, 0.0, 1.0, 1.0],   # Evidence=orig_loc
     [0.0, 1.0, 0.0, 0.0]],  # Evidence=moved_loc
    evidence=['WorldState', 'Observed'],
    evidence_card=[2, 2],
)
# (if not observed, agent's evidence defaults to original location regardless of true world state)

cpd_belief = TabularCPD(
    'Belief', 2,
    [[0.95, 0.05], [0.05, 0.95]],  # Belief tracks Evidence with small noise
    evidence=['Evidence'], evidence_card=[2],
)

cpd_action = TabularCPD(
    'PredictedAction', 2,
    [[0.9, 0.1], [0.1, 0.9]],  # Action follows Belief with small noise
    evidence=['Belief'], evidence_card=[2],
)

model.add_cpds(cpd_world, cpd_observed, cpd_evidence, cpd_belief, cpd_action)
model.check_model()

infer = VariableElimination(model)

class CausalJudgeBN:
    def verdict(self, scenario, claimed_action):
        result = infer.query(
            variables=['PredictedAction'],
            evidence={'Observed': int(scenario.agent_observed_move)},
        )
        p_match = result.values[claimed_action]
        return p_match  # P(claim's asserted action matches the model's predicted action)