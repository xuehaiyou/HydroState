LTAE2d and PositionalEncoder come from VSainteuf/utae-paps, commit
987874e27a98bb399277f2d55eb010744d26f881, under the accompanying MIT license.
HydroState passes a per-pixel padding mask and signed observation days relative
to the requested interval midpoint. Query replication uses tensor expansion.
The experiment replaces the shared temporal pool at the existing encoder exits.
Observation quality still enters state fusion after pooling. This is the L-TAE
module comparison, not an experiment training a full U-TAE from scratch.

The separate UTAE scratch control uses the upstream UTAE spatial encoder,
LTAE bottleneck and all grouped-attention skip decoders. Per-slot channels
contain masked spectra, observation quality and explicit availability. The
output convolution emits 64 features for the project heads. Input positions
are measured in days relative to the center slot. The unrelated recurrent
network in the upstream file is omitted.
