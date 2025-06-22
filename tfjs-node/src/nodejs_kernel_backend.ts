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

import * as tf from '@tensorflow/tfjs';
import { backend_util, DataId, DataType, KernelBackend, ScalarLike, Tensor, Tensor1D, Tensor2D, TensorInfo } from '@tensorflow/tfjs';

import { Int64Scalar } from './int64_tensors';
import { TensorMetadata, TFEOpAttr, TFJSBinding } from './tfjs_binding';

type TensorData = {
  shape: number[],
  dtype: number,
  values: backend_util.BackendValues,
  id: number,
  refCount: number;
};

export class NodeJSKernelBackend extends KernelBackend {
  binding: TFJSBinding;
  isGPUPackage: boolean;
  isUsingGpuDevice: boolean;
  private tensorMap: tf.DataStorage<TensorData>;

  constructor(binding: TFJSBinding, packageName: string) {
    super();
    this.binding = binding;
    this.isGPUPackage = packageName === '@tensorflow/tfjs-node-gpu';
    this.isUsingGpuDevice = this.binding.isUsingGpuDevice();
    this.tensorMap = new tf.DataStorage<TensorData>(this, tf.engine());
  }

  getDTypeInteger(dtype: DataType): number {
    switch (dtype) {
      case 'float32':
        return this.binding.TF_FLOAT;
      case 'int32':
        return this.binding.TF_INT32;
      case 'bool':
        return this.binding.TF_BOOL;
      case 'complex64':
        return this.binding.TF_COMPLEX64;
      case 'string':
        return this.binding.TF_STRING;
      default:
        throw new Error(`Unsupported DType: ${dtype}`);
    }
  }

  // Creates a new Tensor and maps the dataId to the passed in ID.
  private createOutputTensor(metadata: TensorMetadata): Tensor {
    const newId = {};

    this.tensorMap.set(newId, {
      shape: metadata.shape,
      dtype: metadata.dtype,
      id: metadata.id,
      values: null,
      refCount: 1
    });

    let dtype: DataType;
    switch (metadata.dtype) {
      case this.binding.TF_FLOAT:
        dtype = 'float32';
        break;
      case this.binding.TF_INT32:
        dtype = 'int32';
        break;
      case this.binding.TF_INT64:
        console.warn('INT64 output tensor will be stored as BigInt64Array.');
        // INT64 is not supported in TFJS yet, cast it to int32.
        dtype = 'int32';
        break;
      case this.binding.TF_BOOL:
        dtype = 'bool';
        break;
      case this.binding.TF_COMPLEX64:
        dtype = 'complex64';
        break;
      case this.binding.TF_STRING:
        dtype = 'string';
        break;
      case this.binding.TF_RESOURCE:
        // NOTE(cais): We currently represent resource-type Tensors
        // as string of ubytes.
        dtype = 'string';
        break;
      case this.binding.TF_UINT8:
        // TensorFlow uses UINT8 as dtype for image tensor. UINT8 is not
        // supported in TFJS yet, cast it to int32.
        dtype = 'int32';
        break;
      default:
        throw new Error(`Unknown dtype enum ${metadata.dtype}`);
    }

    // TODO(yassogba) Enable this once all the kernels are removed from backend.
    // We can then change the return type from Tensor to TensorInfo.
    // return {dataId: newId, shape: metadata.shape, dtype};

    const tensorInfo:
        TensorInfo = {dataId: newId, shape: metadata.shape, dtype};
    return tf.engine().makeTensorFromTensorInfo(tensorInfo);
  }

  // Prepares Tensor instances for Op execution.
  private getInputTensorIds(tensors: Array<TensorInfo|Int64Scalar>): number[] {
    const ids: number[] = [];
    for (let i = 0; i < tensors.length; i++) {
      if (tensors[i] instanceof Int64Scalar) {
        // Then `tensors[i]` is a Int64Scalar, which we currently represent
        // using an `Int32Array`.
        const value = (tensors[i] as Int64Scalar).valueArray;
        const id = this.binding.createTensor([], this.binding.TF_INT64, value);
        ids.push(id);
      } else {
        const info = this.tensorMap.get((tensors[i] as TensorInfo).dataId);
        // TODO - what about ID in this case? Handle in write()??
        if (info.values != null) {
          // Values were delayed to write into the TensorHandle. Do that before
          // Op execution and clear stored values.
          info.id =
              this.binding.createTensor(info.shape, info.dtype, info.values);
          info.values = null;
        }
        ids.push(info.id);
      }
    }
    return ids;
  }

  createReductionOpAttrs(tensor: TensorInfo, keepDims = false): TFEOpAttr[] {
    return [
      {name: 'keep_dims', type: this.binding.TF_ATTR_BOOL, value: keepDims},
      createTensorsTypeOpAttr('T', tensor.dtype),
      createTensorsTypeOpAttr('Tidx', 'int32')
    ];
  }

  floatPrecision(): 16|32 {
    return 32;
  }

  epsilon(): number {
    return super.epsilon();
  }

  /**
   * Executes an op that has a single input and output.
   *
   * Helper function to wrap executeSingleOutput in a particular case.
   * @param name The name of the Op to execute.
   * @param input The input Tensor for the Op.
   */
  executeSingleInput(name: string, input: TensorInfo): Tensor {
    const opAttrs = [createTensorsTypeOpAttr('T', input.dtype)];
    return this.executeSingleOutput(name, opAttrs, [input]);
  }

  /**
   * Executes a TensorFlow Eager Op that provides one output Tensor.
   * @param name The name of the Op to execute.
   * @param opAttrs The list of Op attributes required to execute.
   * @param inputs The list of input Tensors for the Op.
   * @return A resulting Tensor from Op execution.
   */
  executeSingleOutput(name: string, opAttrs: TFEOpAttr[], inputs: TensorInfo[]):
      Tensor {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), 1);
    return this.createOutputTensor(outputMetadata[0]);
  }

  /**
   * Executes a TensorFlow Eager Op that provides multiple output Tensors.
   * @param name The name of the Op to execute.
   * @param opAttrs The list of Op attributes required to execute.
   * @param inputs The list of input Tensors for the Op.
   * @param numOutputs The number of output Tensors for Op execution.
   * @return A resulting Tensor array from Op execution.
   */
  executeMultipleOutputs(
      name: string, opAttrs: TFEOpAttr[], inputs: TensorInfo[],
      numOutputs: number): Tensor[] {
    const outputMetadata = this.binding.executeOp(
        name, opAttrs, this.getInputTensorIds(inputs), numOutputs);
    return outputMetadata.map(m => this.createOutputTensor(m));
  }

  numDataIds(): number {
    return this.tensorMap.numDataIds();
  }

  dispose(): void {}

  async read(dataId: DataId): Promise<backend_util.BackendValues> {
    return this.readSync(dataId);
  }

  readSync(dataId: DataId): backend_util.BackendValues {
    if (!this.tensorMap.has(dataId)) {
      throw new Error(`Tensor ${dataId} was not registered!`);
    }
    const info = this.tensorMap.get(dataId);
    if (info.values != null) {
      return info.values;
    } else {
      return this.binding.tensorDataSync(info.id);
    }
  }

  /**
   * Dispose the memory if the dataId has 0 refCount. Return true if the memory
   * is released, false otherwise.
   * @param dataId
   * @oaram force Optional, remove the data regardless of refCount
   */
  disposeData(dataId: DataId, force = false): boolean {
    // No-op if already disposed.
    if (this.tensorMap.has(dataId)) {
      const id = this.tensorMap.get(dataId).id;
      this.tensorMap.get(dataId).refCount--;
      if (!force && this.tensorMap.get(dataId).refCount > 0) {
        return false;
      }

      if (id != null && id >= 0) {
        this.binding.deleteTensor(id);
      }
      this.tensorMap.delete(dataId);
    }
    return true;
  }
  /** Return refCount of a `TensorData`. */
  refCount(dataId: DataId): number {
    if (this.tensorMap.has(dataId)) {
      const tensorData = this.tensorMap.get(dataId);
      return tensorData.refCount;
    }
    return 0;
  }

  incRef(dataId: DataId) {
    this.tensorMap.get(dataId).refCount++;
  }

  move(
      dataId: DataId, values: backend_util.BackendValues, shape: number[],
      dtype: DataType, refCount: number): void {
    this.tensorMap.set(
        dataId, {shape, dtype: getTFDType(dtype), values, id: -1, refCount});
  }

  write(values: backend_util.BackendValues, shape: number[], dtype: DataType):
      DataId {
    const dataId = {};
    this.move(dataId, values, shape, dtype, 1);
    return dataId;
  }

  applyActivation<T extends Tensor>(
      input: T, activation: string, preluActivationWeights?: Tensor,
      leakyreluAlpha?: number): T {
    let result = input;
    if (activation != null) {
      if (activation === 'linear') {
        // No-op
      } else if (activation === 'relu') {
        result = tf.relu(result);
      } else if (activation === 'prelu') {
        result = tf.prelu(result, preluActivationWeights) as T;
      } else if (activation === 'leakyrelu') {
        result = tf.leakyRelu(result, leakyreluAlpha);
      } else if (activation === 'elu') {
        result = tf.elu(result);
      } else if (activation === 'relu6') {
        result = tf.relu6(result);
      } else if (activation === 'sigmoid') {
        result = tf.sigmoid(result);
      } else {
        throw new Error(`Activation: ${
            activation} has not been implemented for the Node.js backend`);
      }
    }
    return result;
  }

  divide(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [createTensorsTypeOpAttr(
        'T', backend_util.upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('Div', opAttrs, [a, b]);
  }

  divNoNan(a: Tensor, b: Tensor): Tensor {
    const opAttrs = [createTensorsTypeOpAttr(
        'T', backend_util.upcastType(a.dtype, b.dtype))];
    return this.executeSingleOutput('DivNoNan', opAttrs, [a, b]);
  }

  where(condition: Tensor): Tensor2D {
    return this.executeSingleOutput('Where', [], [condition]) as Tensor2D;
  }

  topKValues<T extends Tensor>(x: T, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  topKIndices(x: Tensor, k: number): Tensor1D {
    throw new Error('Method not implemented.');
  }

  int<T extends Tensor>(x: T): T {
    throw new Error('Method not implemented.');
  }

  getNumOfTFTensors() {
    return this.binding.getNumOfTensors();
  }
}

/** Returns an instance of the Node.js backend. */
export function nodeBackend(): NodeJSKernelBackend {
  return tf.findBackend('tensorflow') as NodeJSKernelBackend;
}

/** Returns the TF dtype for a given DataType. */
export function getTFDType(dataType: tf.DataType): number {
  const binding = nodeBackend().binding;
  switch (dataType) {
    case 'float32':
      return binding.TF_FLOAT;
    case 'int32':
      return binding.TF_INT32;
    case 'bool':
      return binding.TF_BOOL;
    case 'complex64':
      return binding.TF_COMPLEX64;
    case 'string':
      return binding.TF_STRING;
    // tslint:disable-next-line:no-any
    case 'int64' as any:
      // int64 is not a generally supported dtype in TensorFlow.js
      // (tfjs-core). However, it needs to be included here for the purpose of
      // writing the `step` value to TensorBoard via WriteScalarSummary and
      // other op kernels.
      return binding.TF_INT64;
    default:
      const errorMessage = `Unknown dtype: ${dataType}`;
      throw new Error(errorMessage);
  }
}

/**
 * Creates a TFEOpAttr for a 'type' OpDef attribute from a Tensor or list of
 * Tensors.
 */
export function createTensorsTypeOpAttr(
    attrName: string,
    tensorsOrDtype: tf.Tensor|tf.Tensor[]|tf.DataType): TFEOpAttr {
  if (tensorsOrDtype === null || tensorsOrDtype === undefined) {
    throw new Error('Invalid input tensors value.');
  }
  return {
    name: attrName,
    type: nodeBackend().binding.TF_ATTR_TYPE,
    value:
        (tensorsOrDtype instanceof tf.Tensor || Array.isArray(tensorsOrDtype)) ?
        getTFDTypeForInputs(tensorsOrDtype) :
        getTFDType(tensorsOrDtype)
  };
}

// TODO(yassogba) remove? who uses this?
export function createOpAttr(
    attrName: string, tensorsOrDtype: tf.Tensor|tf.Tensor[]|tf.DataType,
    value: ScalarLike): TFEOpAttr {
  if (tensorsOrDtype === null || tensorsOrDtype === undefined) {
    throw new Error('Invalid input tensors value.');
  }
  return {name: attrName, type: nodeBackend().binding.TF_BOOL, value};
}

/** Returns the dtype number for a single or list of input Tensors. */
function getTFDTypeForInputs(tensors: tf.Tensor|tf.Tensor[]): number {
  if (tensors === null || tensors === undefined) {
    throw new Error('Invalid input tensors value.');
  }
  if (Array.isArray(tensors)) {
    for (let i = 0; i < tensors.length; i++) {
      return getTFDType(tensors[i].dtype);
    }
    return -1;
  } else {
    return getTFDType(tensors.dtype);
  }
}

export function ensureTensorflowBackend() {
  tf.util.assert(
      tf.getBackend() === 'tensorflow',
      () => `Expect the current backend to be "tensorflow", but got "${
          tf.getBackend()}"`);
}
