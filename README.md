# Filler-semi-CRF
Codebase for "Transcription free filler word detection with Neural semi-CRFs" [ICASSP2023], arxiv link:

# Setup
This repository requires Python 3.X and Pytorch 1.X. Other packages are listed in requirements.txt.

## Requirements for S4 models
[Reference: Structured State Spaces for Sequence Modeling](https://github.com/HazyResearch/state-spaces#cauchy-kernel)

**Cauchy Kernel**
A core operation of S4 is the "Cauchy kernel" described in the paper. This is actually a very simple operation; a naive implementation of this operation can be found in the standalone in the function cauchy_naive. However, as the paper describes, this has suboptimal memory usage that currently requires a custom kernel to overcome in PyTorch.

Two more efficient methods are supported. The code will automatically detect if either of these is installed and call the appropriate kernel.

**Custom CUDA Kernel**
This version is faster but requires manual compilation for each machine environment. Run python setup.py install from the directory extensions/cauchy/.

# Usage
## Dataset

[PodcastFillers](https://podcastfillers.github.io/) Dataset


# Citations



```bibtex
@article{yan2021skipping,
  title={Skipping the frame-level: Event-based piano transcription with neural semi-crfs},
  author={Yan, Yujia and Cwitkowitz, Frank and Duan, Zhiyao},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={20583--20595},
  year={2021}
}
```

```bibtex
@inproceedings{gu2022efficiently,
  title={Efficiently Modeling Long Sequences with Structured State Spaces},
  author={Gu, Albert and Goel, Karan and R\'e, Christopher},
  booktitle={The International Conference on Learning Representations ({ICLR})},
  year={2022}
}
```

```bibtex
@article{zhu2022filler,
  title={Filler Word Detection and Classification: A Dataset and Benchmark},
  author={Zhu, Ge and Caceres, Juan-Pablo and Salamon, Justin},
  journal={arXiv preprint arXiv:2203.15135},
  year={2022}
}
```