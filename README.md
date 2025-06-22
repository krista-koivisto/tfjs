# TensorNode

TensorNode is a stripped down fork of TensorFlow.js 4.22.0, from which most of
the packages, package building complexity and high-level APIs have been removed,
retaining only the core TensorFlow.js API and the Node.js backend.

This package is not necessarily intended for direct use, although that is
certainly possible. It exists primarily as an interface to hardware for machine
learning tooling.

In other words, this package is primarily intended to exist as a backend to more
user-friendly APIs.

**Develop ML in Node.js** <br/>
Execute native TensorFlow with the TensorNode API under the Node.js runtime.

## About this repo

This repository contains the logic and scripts that combine
several packages.

APIs:
- [TensorFlow.js Core](/tfjs-core),
  a flexible low-level API for neural networks and numerical computation.

Backends/Platforms:
- [TensorFlow.js Node](/tfjs-node), Node.js platform via TensorFlow C++ adapter.
- [TensorFlow.js Node GPU](/tfjs-node-gpu), Node.js platform via TensorFlow C++ adapter with GPU support.

If you care about bundle size, you can import those packages individually.

## Layers

This repository does not include the layers API nor will it be added. This is
intended to provide access to fast TensorFlow C++ kernels, on top of which more
complex utilities can be built.
