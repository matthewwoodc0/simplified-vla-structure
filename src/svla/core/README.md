# Core

`action_space.py` is the canonical policy-action interface. Each representation owns its
vector size, transition encoder, decoder/shape validation, rollout scaling, task executor,
and controller-telemetry normalization. Controller and task implementations remain in their
established modules because moving them would add compatibility risk without changing the
research boundary.
