# --quick: half the loop count of workloads/personalities/log_small_1m.wl.

type log
seed 42
fileset entries 8 size 1048576 prealloc 100 select sequential align 4096

loop 65536:
  open l rw cache lazy
  append l 1048576
  close l

loop 8:
  open l
  readall l
  close l
