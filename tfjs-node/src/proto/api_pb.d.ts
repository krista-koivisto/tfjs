import { Message } from 'google-protobuf';

// This file provides TypeScript typings for the generated `api_pb.js` file.
// The generated JavaScript is produced by `protoc` and therefore uses
// CommonJS `exports = ...` semantics.  In the TypeScript sources we import it
// using a default-import style:
//
//     import messages from './proto/api_pb.js';
//
// To make that work while also getting useful type-checking and IntelliSense
// we declare the shape of the exported object here.  The declarations are
// intentionally restricted to the symbols that are currently used by the
// TypeScript sources inside `tfjs-node` (chiefly in
// `nodejs_kernel_backend.ts`).  More symbols can be added later on an
// as-needed basis.

// tslint:disable:class-name no-namespace
// -----------------------------------------------------------------------------

/* eslint-disable @typescript-eslint/naming-convention */

declare namespace proto.tensorflow {
  // ---------------------------------------------------------------------------
  // Enum definitions
  // ---------------------------------------------------------------------------
  /** TensorFlow data types (subset). */
  enum DataType {
    DT_INVALID = 0,
    DT_FLOAT = 1,
    DT_DOUBLE = 2,
    DT_INT32 = 3,
    DT_UINT8 = 4,
    DT_INT16 = 5,
    DT_INT8 = 6,
    DT_STRING = 7,
    DT_COMPLEX64 = 8,
    DT_INT64 = 9,
    DT_BOOL = 10,
    DT_QINT8 = 11,
    DT_QUINT8 = 12,
    DT_QINT32 = 13,
    DT_BFLOAT16 = 14,
    DT_FLOAT_REF = 101,
    DT_DOUBLE_REF = 102,
    DT_INT32_REF = 103,
    DT_UINT8_REF = 104,
    DT_INT16_REF = 105,
    DT_INT8_REF = 106,
    DT_STRING_REF = 107,
    DT_COMPLEX64_REF = 108,
    DT_INT64_REF = 109,
    DT_BOOL_REF = 110,
    DT_QINT8_REF = 111,
    DT_QUINT8_REF = 112,
    DT_QINT32_REF = 113,
    DT_BFLOAT16_REF = 114
  }

  /** TensorBoard data class enumeration. */
  enum DataClass {
    DATA_CLASS_UNKNOWN = 0,
    DATA_CLASS_SCALAR = 1,
    DATA_CLASS_TENSOR = 2,
    DATA_CLASS_BLOB_SEQUENCE = 3
  }

  // ---------------------------------------------------------------------------
  // Message class declarations (subset)
  // ---------------------------------------------------------------------------

  /** tf.summary histogram plugin metadata. */
  class HistogramPluginData extends Message {
    getVersion(): number;
    setVersion(value: number): HistogramPluginData;
    serializeBinary(): Uint8Array;
  }

  namespace SummaryMetadata {
    /** Nested `PluginData` message. */
    class PluginData extends Message {
      setPluginName(value: string): PluginData;
      setContent(value: Uint8Array): PluginData;
      serializeBinary(): Uint8Array;
    }
  }

  /** Summary metadata top-level message. */
  class SummaryMetadata extends Message {
    setPluginData(value: SummaryMetadata.PluginData): SummaryMetadata;
    setDisplayName(value: string | null): SummaryMetadata;
    setSummaryDescription(value?: string): SummaryMetadata;
    serializeBinary(): Uint8Array;
  }
}

// -----------------------------------------------------------------------------
// The generated JS module attaches everything from `proto.tensorflow` onto the
// CommonJS `exports` object.  That means importing it via a default-import in
// TypeScript gives us an object with the same shape as `proto.tensorflow`.
// -----------------------------------------------------------------------------

declare const messages: typeof proto.tensorflow;
export default messages;
