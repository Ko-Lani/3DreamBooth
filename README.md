<div align="center">
<h1>3DreamBooth</h1>
<h3>High-Fidelity 3D Subject-Driven Video Generation</h3>

<a href="https://arxiv.org/abs/2603.18524"><img src="https://img.shields.io/badge/arXiv-2603.18524-b31b1b.svg?style=flat-square" alt="arXiv"/></a>
<a href="https://ko-lani.github.io/3DreamBooth"><img src="https://img.shields.io/badge/Project-Page-DAA520?style=flat-square" alt="Project Page"/></a>



https://github.com/user-attachments/assets/f968f978-01ab-4e78-98be-cbe212d44d1a


</div>

---

## Overview

**3DreamBooth** is a novel framework for high-fidelity 3D subject-driven video generation. Given a set of multi-view reference images of a subject, our method generates identity-preserving, view-consistent videos with rich 3D spatial awareness.

Our framework comprises two components:

- **3Dapter** — a visual conditioning module that enhances fine-grained texture preservation and accelerates convergence via multi-view joint attention with shared weights.
- **3DreamBooth** — a DreamBooth-style test-time optimization that decouples spatial geometry from temporal motion through a 1-frame optimization paradigm, baking a robust 3D prior without exhaustive video-based training.

---

## Results

3DreamBooth generates cinematic, identity-preserving videos across diverse subjects and creative scenarios — bags, plushies, sculptures, motorcycles, watches, and more.

> 📽️ See our [project page](https://ko-lani.github.io/3DreamBooth) for full video results.

---

## Code Release

We are in the process of cleaning up the code for public release.

- [ ] Inference code
- [ ] 3Dapter pretrained weights
- [ ] 3D-CustomBench benchmark
- [ ] Training code

**Star ⭐ this repo to get notified when the code drops.**

---

## Citation

If you find our work useful, please consider citing:

```bibtex
@misc{ko20263dreamboothhighfidelity3dsubjectdriven,
  title         = {3DreamBooth: High-Fidelity 3D Subject-Driven Video Generation Model},
  author        = {Hyun-kyu Ko and Jihyeon Park and Younghyun Kim and Dongheok Park and Eunbyung Park},
  year          = {2026},
  eprint        = {2603.18524},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.18524},
}
```
