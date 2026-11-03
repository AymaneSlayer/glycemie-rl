from .controllers import PID
from .features import raw_features
from .simulator import Simulator


def run_episode(controller, patient, seed, meals=None, with_features=False):
    sim = Simulator(patient, seed=seed, meals=meals)
    controller.reset(sim)
    shadow = None
    if with_features and not hasattr(controller, "pid"):
        shadow = PID()
        shadow.reset(sim)
    feats = []
    while not sim.done:
        basal = controller.act(sim)
        if with_features:
            feats.append(raw_features(sim, shadow.act(sim)) if shadow else controller.last_features)
        sim.step(basal, controller.bolus_factor())
    df = sim.history()
    df["controller"] = controller.name
    if with_features:
        return df, feats
    return df
