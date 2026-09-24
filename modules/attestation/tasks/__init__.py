"""attestation module tasks -- SEV-SNP attestation verified by Veraison."""


def register(ns):
    from invoke import Collection

    from core.tasks.actions.module import register_module
    from modules.attestation.tasks import attestation, veraison

    register_module(
        ns, actions="modules.attestation.tasks.actions", commands=attestation
    )
    ns.add_collection(Collection.from_module(veraison))
