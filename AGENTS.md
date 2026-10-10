# Repository rules

- Keep changes minimal and additive to upstream Pixal3D. New published functionality is inference only; preserve existing behavior unless a feature explicitly changes it.
- Keep our training, evaluation, tests, data preparation, experiments and machine configuration under the Git-ignored `.research/` directory. Public inference must never import from it. Do not stage assets, weights, logs or private configuration.
- Public code must be portable: use arguments, standard cache locations or documented environment variables. Never hardcode usernames, machine paths, SSH hosts, ports or private services.
- Keep one public README focused on installation, required weights, inference examples and supported behavior. Keep research plans and run status outside the public README.
- Commit by complete functional unit, with the smallest coherent diff. Before committing, inspect the staged diff and verify that no private files or machine details are included. Incremental commits are authorized; do not push unfinished features or force-push.
- Verify checkpoint compatibility, static-input identity, causal isolation and temporal padding using private tests. Publish capabilities only after end-to-end inference validation.
