#!/usr/bin/env python
import os
import subprocess


def get_git_rev():
    try:
        new_cwd = os.path.abspath(os.path.dirname(__file__))
        rev_raw = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=new_cwd)
        return rev_raw.decode("utf8").strip()
    except Exception:
        return "unknown"


def mkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_baked_revision(base_dir):
    dest_dir = os.path.join(base_dir, '..', 'build')
    baked_dest = os.path.join(dest_dir, '_baked_revision.py')
    mkpath(dest_dir)

    with open(baked_dest, "w") as f:
        f.write(r'revision = "%s"' % get_git_rev())


if __name__ == "__main__":
    here = os.path.abspath(os.path.dirname(__file__))
    write_baked_revision(here)
