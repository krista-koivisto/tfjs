# How to Contribute

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

Please follow the guidelines specified in the
[ISSUE_TEMPLATE](https://github.com/tensorflow/tfjs/tree/master/.github/ISSUE_TEMPLATE)
file.

## Adding functionality

One way to ensure that your PR will be accepted is to add functionality that
has been requested in Github issues. If there is something you think is
important and we're missing it but does not show up in Github issues, it would
be good to file an issue there first so we can have the discussion before
sending us a PR.

In general, we're trying to add functionality when driven by use-cases instead of
adding functionality for the sake of parity with TensorFlow. This will help us
have less to maintain.

### Adding an op

When adding ops to the library and deciding whether to write a kernel
implementation in [backend.ts](/tfjs-core/src/backends/backend.ts),
be sure to check out the TensorFlow ops list [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/ops.pbtxt).
This list shows the kernels available for the TensorFlow C API. To ensure that
we can bind to this with Node.js, we should ensure that our `backend.ts`
interface matches ops in the TensorFlow C API.
