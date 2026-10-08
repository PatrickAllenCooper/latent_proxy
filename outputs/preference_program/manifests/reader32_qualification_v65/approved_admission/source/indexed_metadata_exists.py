"""Single-threaded, bounded-use accelerator for original metadata enumeration.
No mappings fabricated: run original packages_distributions with equivalent Path.exists.
Not safe with concurrent filesystem mutation/imports; fail closed on directory change.
"""
import os
from pathlib import Path

class DirectoryChanged(RuntimeError):pass

class ExistsIndex:
    def __init__(self, original_exists=Path.exists):
        self.original_exists=original_exists
        self.directories={}
        self.lookups=0
        self.fallbacks=0
    @staticmethod
    def signature(path):
        st=os.stat(path)
        return (st.st_dev,st.st_ino,st.st_mtime_ns,st.st_ctime_ns)
    def exists(self,path,*args,**kwargs):
        self.lookups+=1
        path=Path(path)
        # Preserve optional follow_symlinks semantics and special final components.
        if args or kwargs or path.name in ('','.', '..'):
            self.fallbacks+=1;return self.original_exists(path,*args,**kwargs)
        parent=path.parent
        if parent not in self.directories:
            try:
                before=self.signature(parent)
                with os.scandir(parent) as entries:
                    names={e.name:e.is_symlink() for e in entries}
                after=self.signature(parent)
            except OSError:
                self.fallbacks+=1;return self.original_exists(path)
            if before!=after:raise DirectoryChanged(str(parent))
            self.directories[parent]=(after,names)
        _,names=self.directories[parent]
        if path.name not in names:return False
        if names[path.name]:
            # A name alone does not prove a live symlink target.
            self.fallbacks+=1;return self.original_exists(path)
        return True
    def validate(self):
        for parent,(signature,_) in self.directories.items():
            if self.signature(parent)!=signature:raise DirectoryChanged(str(parent))
    def receipt(self):
        return {'lookups':self.lookups,'fallbacks':self.fallbacks,'indexed_directories':len(self.directories),'directory_signatures':{str(p):list(v[0]) for p,v in self.directories.items()}}

def capture_original_mapping(metadata_module):
    """Original mapping algorithm and distribution/version metadata remain unchanged."""
    original=Path.exists;index=ExistsIndex(original)
    def exists(path,*args,**kwargs):return index.exists(path,*args,**kwargs)
    Path.exists=exists
    try:
        mapping=metadata_module.packages_distributions()
        index.validate()
        return mapping,index.receipt()
    finally:Path.exists=original
