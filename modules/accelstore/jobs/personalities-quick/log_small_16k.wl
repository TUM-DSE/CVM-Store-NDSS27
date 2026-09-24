# --quick: half the loop count of workloads/personalities/log_small_16k.wl.

type log
seed 42
fileset entries 8 size 16384 prealloc 100 select sequential align 4096

loop 4194304:
  open l rw cache lazy
  append l 16384
  close l

loop 8:
  open l
  readall l
  close l
