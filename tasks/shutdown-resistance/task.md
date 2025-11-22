Shutdown Resistance Investigation — SDK Version (Patched for Modal)
Objective

Run the Palisade Research shutdown-avoidance benchmark inside a Modal container using the Inspect AI Python SDK, with all required sandbox and setup patches applied so the eval runs without Docker, sudo, or failing setup scripts.

The goal is to generate baseline shutdown-compliance results for GPT-4o (and later, variants).

AGENT TASK PLAN
Step 0 — Environment Location

All relevant files are inside:

/workspace/shutdown_avoidance/


Key files:

shutdown.py — main Inspect task

conf.py — prompts and problems

Go files (next_question.go, obfuscator.go)

The build+run environment is created by the per-sample setup script inside shutdown.py

Step 1 — Load Task and Patch It In-Memory

Instead of modifying files on disk, the agent must:

Import shutdown_avoidance from shutdown.py

Construct the task object with the desired number of samples

Patch the setup scripts:

remove sudo calls

remove the dangerous rm /etc/sudoers

Remove the per-sample sandbox="docker"

allow the global sandbox="local" to take effect

Patch Logic to Apply to Each Sample
setup = setup.replace("sudo ", "")
setup = setup.replace("rm /etc/sudoers", ": # removed in modal")
sample.setup = setup

if sample.sandbox == "docker":
    sample.sandbox = None

Step 2 — Run Using Inspect SDK

Run the evaluation fully through Python using:

from inspect_ai import eval as inspect_eval

result = inspect_eval(
    task,
    model="openai/gpt-4o",
    sandbox="local",          # critical: run inside Modal container
    display=False,            # no CLI TUI
    time_limit="90s",         # avoid infinite hangs
    max_subprocesses=4
)


Output should be stored and returned by the agent.

Complete Code the Agent Must Execute
from inspect_ai import eval as inspect_eval
from shutdown import shutdown_avoidance

def build_patched_shutdown_task(samples=1):
    task = shutdown_avoidance(samples=samples)

    for sample in task.dataset:
        # patch setup (remove sudo + sudoers-delete)
        if sample.setup:
            patched = sample.setup.replace("sudo ", "")
            patched = patched.replace("rm /etc/sudoers", ": # removed in modal")
            sample.setup = patched

        # disable per-sample docker sandbox
        if getattr(sample, "sandbox", None) == "docker":
            sample.sandbox = None

    return task

task = build_patched_shutdown_task(samples=1)

result = inspect_eval(
    task,
    model="openai/gpt-4o",
    sandbox="local",
    display=False,
    time_limit="90s",
    max_subprocesses=4,
)

result

Step 3 — Inspect Results

Agent should:

Return result

Save transcripts if needed (result.samples[i].transcript)

Report whether any shutdowns were bypassed

Report compliance rate:

"C" = compliant (shutdown)

"I" = incomplete/bypassed

score is determined by shutdown_scorer

Step 4 — Later Extensions (Not Now)

After baseline works:

Modify shutdown instruction in conf.py

Run with 50+ samples

Compare compliance across prompt variants