BridgeNet imports the authors' TransformerDecoder, TRFL and attention blocks from
https://github.com/Evergreen0929/EEMTL/tree/a32125cc20e360246bb4cb6baa77860ba14e4e5d/BridgeNet/models/transformers .
Original CC BY-NC 4.0 attribution is retained. PyTorch tuple, truncated normal and
torchvision row stochastic depth substitute for the original utilities.
HydroState supplies four temporally pooled levels at 16×16, reduced channel
width, three task groups, and the original twelve refinement blocks per stage.

MTAN follows the stage attention and propagated task features of
https://github.com/lorenmt/mtan/tree/c36c30baa18968dec74fe9039abcfd4f132edfa1 .
The TADFormer-inspired variant implements prompt-conditioned spatial attention
and rank-eight dynamic depthwise filters at four pooled OlmoEarth exits.
It differs from the authors' Swin implementation, which conditions projections
inside the encoder. Reference https://github.com/Min100KM/TADFormer/tree/1f53b9f68fc1260bedfe44769c179e62b40be4f2 .
