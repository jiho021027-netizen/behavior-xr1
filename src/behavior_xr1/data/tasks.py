from __future__ import annotations
class TaskMap:
 def __init__(self, tasks):
  self._tasks=tuple(str(x) for x in tasks)
  if len(self._tasks)!=100 or len(set(self._tasks))!=100: raise ValueError('task map must contain exactly 100 unique task strings')
  self._ids={x:i for i,x in enumerate(self._tasks)}
 def id_for(self, task):
  if task not in self._ids: raise KeyError(task)
  return self._ids[task]
 def task_for(self, idx):
  if not isinstance(idx,int) or idx<0 or idx>=100: raise KeyError(idx)
  return self._tasks[idx]
 def __len__(self): return 100
