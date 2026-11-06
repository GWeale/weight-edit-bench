# Security and isolation

This repository provides bounded interpreters for student code and tensor-edit
programs. They reject unsupported operations and cap work. They are not an
operating-system sandbox or a reviewed security boundary for hostile agents.

An in-process editor can inspect Python objects and local files. For adversarial
evaluation, run the agent in a separate restricted process or container with no
dataset, checkpoint, credential or host-filesystem access. Expose only validated
actions and public observations through a narrow transport. Container isolation
and a remote execution service are not supplied by this release.

The local model loader uses pinned manifests and disables model remote code.
Review the source and license of models you download. Never run arbitrary Python
emitted by an editor in place of the restricted tensor language. Resource caps
do not guarantee that every permitted program fits every device's RAM.

Do not put secrets or raw private prompts in a public issue. Use GitHub private
vulnerability reporting when it is enabled on this repository. Otherwise report
the affected component without sensitive exploit material and request a private
reporting channel.
