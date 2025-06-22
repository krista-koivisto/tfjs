/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

// Register all kernels.
import './register_all_kernels';

import * as tf from '@tensorflow/tfjs';
import * as path from 'path';

import * as nodeIo from './io/index';
import { NodeJSKernelBackend } from './nodejs_kernel_backend';
import { TFJSBinding } from './tfjs_binding';

// Get the package name from the package.json file.
import { name as packageName } from '../package.json';

import binary from '@mapbox/node-pre-gyp';
import fs from 'fs';

const bindingPath =
    binary.find(path.resolve(path.join(__dirname, '/../package.json')));

// Check if the node native addon module exists.

if (!fs.existsSync(bindingPath)) {
  throw new Error(
      `The Node.js native addon module (tfjs_binding.node) can not ` +
      `be found at path: ` + String(bindingPath) + `. \nPlease run command ` +
      `'npm rebuild @tensorflow/tfjs-node` +
      (String(bindingPath).indexOf('tfjs-node-gpu') > 0 ? `-gpu` : ``) +
      ` --build-addon-from-source' to ` +
      `rebuild the native addon module. \nIf you have problem with building ` +
      `the addon module, please check ` +
      `https://github.com/tensorflow/tfjs/blob/master/tfjs-node/` +
      `WINDOWS_TROUBLESHOOTING.md or file an issue.`);
}

/**
 * TODO: Can we get rid of this `require`?
 *
 * ES-module import statements have to be static and resolve at compile time,
 * but bindingPath is computed at runtime.
 *
 * Node’s dynamic import() returns a Promise and, as of today, does not reliably
 * load native .node addons.
 *
 * require() is the officially supported way.
 */
const bindings = require(bindingPath);

export const io = {
  ...tf.io,
  ...nodeIo
};

// Export all union package symbols
export * from '@tensorflow/tfjs';


// Side effects for default initialization of Node backend.
tf.registerBackend('tensorflow', () => {
  return new NodeJSKernelBackend(bindings as TFJSBinding, packageName);
}, 3 /* priority */);

const success = tf.setBackend('tensorflow');
if (!success) {
  throw new Error(`Could not initialize TensorFlow backend.`);
}
