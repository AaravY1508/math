"""Maximum Fence Area research harness (Erich Friedman's packing problem).

Modules
-------
geometry  float64 topology/DCEL + batched differentiable residuals (optimization path)
referee   independent 50-digit mpmath verifier -- the only code that certifies results
builder   sketch-construction DSL used by evolved `propose_topology` programs
optimize  batched GPU Adam with annealed penalties + CPU SLSQP polish + verification
evolve    island-model MAP-Elites evolution of proposer programs (random / llm mutation)
"""
