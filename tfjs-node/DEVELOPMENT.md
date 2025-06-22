# TensorNode Node.js bindings development.

The @tensorflow/tfjs-node repo supports npm package @tensorflow/tfjs-node and @tensorflow/tfjs-node-gpu on Windows/Mac/Linux. This guide lists commands to use when developing this package.

## Install

#### Dependencies and addon module

```sh
$ yarn
```

This command installs all dependencies and devDependencies listed in package.json. It also downloads the TensorFlow C library and native node addon.

#### Compile native addon from source files

```sh
$ yarn build:addon
```

This command will compile a new native node addon from source files.

####

```sh
$ yarn install
```

This command does the following:

1. Clears local binary and addon resources
2. Downloads the TensorFlow C library
3. Compiles the native addon from source files (instead of downloading pre-compile addon)

#### Build package

```sh
$ yarn build
```
