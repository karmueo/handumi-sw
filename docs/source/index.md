# HandUMI

Collect robot-free bimanual demonstrations once with HandUMI, then validate,
retarget, and reuse them across different bimanual arms with parallel grippers.

**Core Workflows** need no robot arm: record with HandUMI, preview an
embodiment in simulation, then replay and validate the result.
**Physical Robots** covers everything that commands real hardware.

```{image} _static/handumi-overview.png
:alt: HandUMI overview. Left, an operator wearing two HandUMI devices, the labelled device with fisheye camera, VR controller, servo encoder and swappable fingertips, and the PICO 4 Ultra and Meta Quest headsets. Right, recorded trajectories retargeted to seven bimanual arms.
:class: handumi-cover
:width: 100%
```

## Paper

**HandUMI: A Modular Hand-Worn Parallel-Jaw Interface for Robot-Free Bimanual
Data Capture**  
Brik Meza, Leonardo Perez, Alvaro Mendoza, Aniela Ubillus, Mitshell Ramos,
Raul Bastidas, Mauricio Olivera, Bryan Bastidas

[Project website](https://murobotics.ai/research/handumi) ·
[Hardware](https://github.com/murobotics-ai/handumi-hw) ·
[Software](https://github.com/murobotics-ai/handumi-sw) ·
[Quest app](https://github.com/murobotics-ai/handumi-quest-app) ·
[Data and checkpoints](https://huggingface.co/collections/murobotics/handumi-towel-folding)

HandUMI's swappable fingertips match the robot's installed parallel-jaw
gripper, so the robot keeps its gripper. Each demonstration is stored once in
robot-agnostic form and retargeted by inverse kinematics to seven bimanual
arms. On bimanual towel folding, 562 robot-free demonstrations (111.9 min, no
robot-collected data) train policies that fold the towel on real AgileX PiPER
arms: ACT succeeds in 12/20 attempts and fine-tuned π0.5 in 7/20, while ACT
trained on the first 101 demonstrations and π0.5 without fine-tuning succeed
in none.

```{image} _static/towel-folding-results.png
:alt: Towel-folding task. Left and right fisheye wrist views, the ZED Mini top view, a PiPER rollout from initial state to grasping edges to folding the towel, and real-robot success rates (ACT 101 episodes 0/20, ACT 562 episodes 12/20, pi0.5 base 0/20, pi0.5 562 episodes 7/20).
:width: 100%
```

If you use HandUMI, please cite:

```bibtex
@misc{meza2026handumi,
  title  = {{HandUMI}: A Modular Hand-Worn Parallel-Jaw Interface for
            Robot-Free Bimanual Data Capture},
  author = {Meza, Brik and Perez, Leonardo and Mendoza, Alvaro and
            Ubillus, Aniela and Ramos, Mitshell and Bastidas, Raul and
            Olivera, Mauricio and Bastidas, Bryan},
  year   = {2026},
  url    = {https://murobotics.ai/research/handumi}
}
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting_started/installation
setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Core Workflows

record
teleoperation
workflows/replay_in_sim
workflows/franka_emika_panda
workflows/vive_umi_rm75b
workflows/datasets
workflows/context_inpainting
workflows/dataset_curation
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Physical Robots

physical_robots/real_teleoperation
physical_robots/piper_setup
physical_robots/openarm_v1_setup
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Help

troubleshooting
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Development

development/new_embodiment
```
