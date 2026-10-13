import argparse, torch
from omnigibson.eval.utils.network_utils import WebsocketPolicyServer
from behavior_xr1.evaluation.mock_xr1 import neutral_bridge
class Policy:
 def reset(self): pass
 def act(self,obs):
  print('WS_REQUEST_BEGIN',flush=True); print('STATE_BRIDGE_BEGIN',flush=True); state,action,out=neutral_bridge(obs); print('STATE_BRIDGE_PASS',flush=True);print('XR1_STATE_SHAPE='+str(state.shape[-1]),flush=True);print('XR1_STATE_FINITE='+str(bool(torch.isfinite(torch.tensor(state)).all())),flush=True);print('XR1_RESERVED_TAIL_ZERO='+str(bool((state[...,16:]==0).all())),flush=True);print('MOCK_POLICY_PASS',flush=True);print('XR1_ACTION_SHAPE=60',flush=True);print('ACTION_BRIDGE_PASS',flush=True);print('BRIDGED_ACTION_SHAPE=21',flush=True);print('ACTION_SENT',flush=True);return torch.tensor(out,dtype=torch.float32)
p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8000);a=p.parse_args();WebsocketPolicyServer(Policy(),host=a.host,port=a.port,metadata={'behavior_xr1_action_dim':21,'mode':'mock-xr1-neutral'}).serve_forever()
