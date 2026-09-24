"""accelstore module tasks — AccelStore client benchmarks and paper plotting."""


def register(ns):
    """Register accelstore module into the given Collection."""
    from invoke import Collection

    from core.tasks.actions.module import register_module
    from modules.accelstore.tasks import accelstore, paper

    register_module(ns, actions="modules.accelstore.tasks.actions", commands=accelstore)
    ns.add_collection(Collection.from_module(paper))
