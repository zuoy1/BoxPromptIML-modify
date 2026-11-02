"""Process audit of dataset file reads; rejects all non-student image paths."""
import os
from pathlib import Path
import sys


class DataReadAudit:
    def __init__(self, rgb_paths, cache_paths):
        self.rgb={str(Path(p).resolve()) for p in rgb_paths}
        self.cache={str(Path(p).resolve()) for p in cache_paths}
        self.opened=[]
        self.denied=[]
        self.active=False

    def _event(self,event,args):
        if not self.active or event!='open' or isinstance(args[0],int):
            return
        path=str(Path(os.fsdecode(args[0])).resolve())
        if not (path.startswith('/home/datasets/') or '/stage2c_cache/' in path):
            return
        mode=args[1] or ''
        flags=args[2] or 0
        writing=any(c in mode for c in ('w','a','+','x')) or flags&(
            os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)
        if path not in self.rgb|self.cache or writing:
            self.denied.append(path)
            raise PermissionError('Training read outside frozen student RGB/cache allowlist')
        self.opened.append(path)

    def __enter__(self):
        sys.addaudithook(self._event)
        self.active=True
        return self

    def __exit__(self,*exc):
        self.active=False
