from .features import raw_features
from .simulator import Simulator


def run_episode(controller, patient, seed, meals=None, with_features=False):
    sim = Simulator(patient, seed=seed, meals=meals)
    controller.reset(sim)
    feats = []
    while not sim.done:
        if with_features:
            feats.append(raw_features(sim))
        sim.step(controller.act(sim))
    df = sim.history()
    df["controller"] = controller.name
    if with_features:
        return df, feats
    return df
