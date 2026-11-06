# Security and isolation

This repository provides bounded interpreters for student code and tensor-edit
programs. They reject unsupported operations and cap work. They are not an
operating-system sandbox or a reviewed security boundary for hostile agents.

An in-process editor can inspect Python objects and local files. The Harbor route
supplies an agent container and a separate verifier container. The verifier starts
from the original image, reads its trusted task specification, and accepts only
a bounded JSON transcript at `/app/submission.json`. Agent changes to its Python
library, local model, specification or printed scores are not grading authority.
The verifier uses isolated Python imports and does not execute agent-authored
Python modules. See the [Harbor contract](docs/public/harbor.md).

This separation protects grading from ordinary agent-container modifications;
it is not an audited defense against container escape. Do not expose the host
Docker socket, credentials or unrelated host files to agents. Public generators,
seeds and source are available inside tasks, so agents can reconstruct final
data. This release does not provide secret evaluation data or a hosted service.
Harbor's wall-time/resource limits bound execution, while transcript counters
exclude discarded exploratory attempts.

The local model loader uses pinned manifests and disables model remote code.
Review the source and license of models you download. Never run arbitrary Python
emitted by an editor in place of the restricted tensor language. Resource caps
do not guarantee that every permitted program fits every device's RAM.

Do not put secrets or raw private prompts in a public issue. Use GitHub private
vulnerability reporting when it is enabled on this repository. Otherwise report
the affected component without sensitive exploit material and request a private
reporting channel.
