from __future__ import annotations
import argparse, numpy as np, torch
from omnigibson.eval.utils.network_utils import WebsocketPolicyServer
class DummyPolicy:
 def __init__(self, action_dim=23): self.action_dim=action_dim
 def reset(self): pass
 def act(self, obs): return torch.zeros(self.action_dim,dtype=torch.float32)

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--policy',choices=['dummy','xr1'],default='dummy'); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=8000); a=p.parse_args(argv)
 if a.policy!='dummy': raise RuntimeError('XR-1 policy backend is unavailable until a local checkpoint is explicitly configured')
 WebsocketPolicyServer(DummyPolicy(),host=a.host,port=a.port,metadata={'behavior_xr1_policy':a.policy}).serve_forever()
if __name__=='__main__': main()
