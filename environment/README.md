# Execution environments

The original calculations were executed in separate Python environments
for different pretrained interatomic-potential packages because several
model families require incompatible dependency stacks.

Environment directory names appearing in the executed scripts, such as
`.pilot-venv`, `.orb-venv`, `.grace-venv`, `.esen-venv`,
`.eqv2-venv`, `.sevennet-venv`, and `.pet-venv`, identify the local
environments used for the reported calculations.

The virtual-environment directories themselves are not distributed.
Users should install the corresponding model packages and dependencies
using the versions documented with the reproducibility release.

Pretrained model checkpoints are also not redistributed. They should be
obtained from their original providers and placed at the relative paths
expected by the corresponding inference scripts.
