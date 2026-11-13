# Provenance

- Paper/project: [Hiking in the Wild](https://project-instinct.github.io/hiking-in-the-wild/), Project Instinct.
- Environment source snapshot: local `agibot_x2_hiking_in_the_wild` fork of [InstinctLab](https://github.com/project-instinct/InstinctLab), commit `411779fe810f0cb021e177b63ad48d9e15258f05`. The G1 parkour configuration, motor constants, rewards, terrains, target command, edge consolidation and cylinder grid were retained/adapted. Original source and license are in `vendor/InstinctLab`, pinned as a submodule of the public [X2 fork](https://github.com/Nagato-Yukii/agibot_x2_hiking_in_the_wild). The local image-only modification is not included in this submodule pin.
- Algorithm: local [instinct_rl](https://github.com/project-instinct/instinct_rl), commit `ba45ed231ebbf0a4099cd31d607e2886814fd165`, copied under `vendor/instinct_rl`. The original WasabiPPO algorithm, depth encoder and MoE networks are used. Preserve its root license and inherited file-level notices.
- InstinctLab and instinct_rl root licenses are **CC BY-NC 4.0**; inherited NVIDIA/ETH/Isaac Lab portions have their own notices. This port does not relicense them.
- Isaac Lab: official `v3.0.0`, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`. Copied camera update/kernel and legacy height-field decorator retain BSD-3-Clause attribution.
- Motion release: official project-page Google Drive link `10tQylYHdKLVDVnmVHrygLB70neHnPdoF`. Source URL and locally measured SHA-256 values are recorded in `data/provenance.json`.

The G1 parkour path runs on Isaac Lab 3.0 / Newton. The vendored source trees retain the additional upstream task implementations.
