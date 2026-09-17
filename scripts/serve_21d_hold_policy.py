from __future__ import annotations

import argparse
import torch

from omnigibson.eval.utils.network_utils import WebsocketPolicyServer


class Hold21DPolicy:
    """Shape-only diagnostic policy for the official websocket protocol."""

    def reset(self):
        return None

    def act(self, obs):
        action = torch.zeros(21, dtype=torch.float32)
        print(f"POLICY_ACTION_SHAPE={tuple(action.shape)}", flush=True)
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    WebsocketPolicyServer(
        Hold21DPolicy(), host=args.host, port=args.port,
        metadata={"behavior_xr1_action_dim": 21, "mode": "hold-diagnostic"},
    ).serve_forever()


if __name__ == "__main__":
    main()
