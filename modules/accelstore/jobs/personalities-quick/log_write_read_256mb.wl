# --quick: half the loop count of workloads/personalities/log_write_read_256mb.wl.

type log
seed 42
fileset entries 8 size 268435456 prealloc 100 select sequential align 4096

loop 512:
  open l rw cache lazy
  append l 268435456
  close l

loop 8:
  open l
  readall l
  close l
