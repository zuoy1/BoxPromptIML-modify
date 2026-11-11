"""区域实验进程的数据allowlist与历史写保护。"""
import os
from pathlib import Path
import sys


class RegionReadAudit:
    def __init__(self, rgb, cache, output):
        self.allowed = {str(Path(p).resolve()) for p in list(rgb)+list(cache)}
        self.output = Path(output).resolve()
        self.opened, self.denied = [], []
        self.active = False

    def _event(self, event, args):
        if not self.active or event != 'open' or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        writing = writing or bool(mode and any(c in mode for c in 'wax+'))
        text = str(path)
        data = (text.startswith(('/home/datasets/', '/hy-tmp/datasets/'))
                or '/fecd/assets/' in text or '/fecd_runs/' in text)
        denied = ((writing and not path.is_relative_to(self.output))
                  or (not writing and data and text not in self.allowed
                      and not path.is_relative_to(self.output)))
        if denied:
            self.denied.append(text)
            raise PermissionError('区域实验访问边界拒绝: '+text)
        if text in self.allowed:
            self.opened.append(text)

    def __enter__(self):
        sys.addaudithook(self._event)
        self.active = True
        return self

    def __exit__(self, *exc):
        self.active = False
