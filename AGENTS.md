# Repository rules

- Keep changes minimal and additive to upstream Pixal3D. Publish inference and CSV-driven evaluation; preserve existing behavior unless a feature explicitly changes it.
- Keep our training, tests, data preparation, experiments and machine configuration under the Git-ignored `.research/` directory. Public code must never import from it. Do not stage assets, weights, logs or private configuration.
- Public code must be portable: use arguments, standard cache locations or documented environment variables. Never hardcode usernames, machine paths, SSH hosts, ports or private services.
- Keep one public README focused on installation, required weights, inference, evaluation protocols and supported behavior. Keep research plans and run status outside the public README.
- Commit by complete functional unit, with the smallest coherent diff. Before committing, inspect the staged diff and verify that no private files or machine details are included. Incremental commits are authorized; do not push unfinished features or force-push.
- The temporal-adapter route is deprecated on `pixal4d/development`; new work uses `pixal4d/continuous-motion`. Keep native Pixal3D frozen and preserve per-frame geometry/features. Do not revive old adapter training implicitly.
- Verify physical time units, shared spatial coordinates, sparse support, static identity, bidirectional integration and gradients using private tests. Distinguish mathematical primitives, oracle geometry diagnostics and learned video inference; publish capabilities only after their corresponding validation. Never infer material-point identity from voxel row order.
