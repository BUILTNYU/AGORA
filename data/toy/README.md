# Toy network (Section 4.1)

`toynetwork.csv` is a sectioned CSV: each block starts with `# SECTION: <name>`,
then a header row, then data rows; a blank line ends the block. Sections are
SCALARS, OPERATORS, LINKS, PATHS.

Three nodes — suburb (1), hub (2), downtown (3) — and two operators. BLUE
(commuter bus, F = 1200) owns l1 and l2; RED (on-demand, F = 250) owns l3 and
l4. Links l5–l7 are outside options in A_0 and carry Phi = 0. Scales are
alpha_B = 1.0, alpha_S = 0.8; the three link features are distance, in-vehicle
time and out-of-vehicle time with theta = (1.20, -0.25, -0.50).

The same instance is constructed in code by `algorithm1_p1.toy_instance()` and
`algorithm2_p2.toy_instance_p2()`, which is what `verify_toy.py` runs. The CSV
is the readable statement of the inputs; the code is what the paper's numbers
come from.
