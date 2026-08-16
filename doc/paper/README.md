# Apple Metal decoder characterization paper

`apple_metal_decoder_characterization.tex` is a two-column experimental
systems paper covering the implementation, measurement protocol, latency,
cold start, stage costs, crossover, memory, process-attributed energy,
correctness, perceptual behavior, and threats to validity.

Build it without leaving generated files in the source tree:

```sh
paper_dir="$PWD/doc/paper"
paper_build="$(mktemp -d /tmp/jpegli-paper.XXXXXX)"
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error \
  -outdir="$paper_build" \
  "$paper_dir/apple_metal_decoder_characterization.tex"
```

The rounded values in the paper come from `doc/apple_metal_results.md` and
`doc/apple_energy_results.md`. The checksum-verified experiment artifact also
contains the raw latency, crossover, and energy CSV files, test logs, exact
source revision, binaries, installed headers, and host metadata.

The paper calls `ri_energy_nj` process-attributed energy. It must not be
relabeled as whole-device, battery, adapter, or rail energy.
