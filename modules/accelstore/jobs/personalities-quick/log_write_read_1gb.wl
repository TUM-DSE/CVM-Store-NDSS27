# --quick: half the loop count of workloads/personalities/log_write_read_1gb.wl.

type log
seed 42
fileset entries 8 size 1073741824 prealloc 100 select sequential align 4096

loop 128:
  open l rw cache lazy
  append l 1073741824
  close l

loop 8:
  open l
  readall l
  close l
