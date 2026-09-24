# Hack to make libpyaccelstore.so work if non NixOS python  
{ runCommand, lib, patchelf, pyaccelstore, imageDir ? "/opt/accelstore/lib" }:

let
  collidingSonames = [ "libcrypto.so.3" "libssl.so.3" ];

  systemSonames = [
    "libc.so.6"
    "libm.so.6"
    "libdl.so.2"
    "librt.so.1"
    "libpthread.so.0"
    "libresolv.so.2"
    "libutil.so.1"
    "ld-linux-x86-64.so.2"
  ];
in
runCommand "accelstore-isolated-bridge"
{
  nativeBuildInputs = [ patchelf ];
  passthru = { inherit imageDir; library = "${imageDir}/libpyaccelstore.so"; };
} ''
  set -euo pipefail
  dest=$out${imageDir}
  mkdir -p "$dest"

  system=" ${lib.concatStringsSep " " systemSonames} "

  needed_of() { patchelf --print-needed "$1" | tr '\n' ' '; }

  # DT_NEEDED resolved through each object's own RUNPATH, which is what the
  # loader does; ldd would report the build machine's resolution instead.
  declare -A resolved=()
  queue=( "${pyaccelstore.library}" )
  while [ ''${#queue[@]} -gt 0 ]; do
    current=''${queue[0]}
    queue=( "''${queue[@]:1}" )
    rpath=$(patchelf --print-rpath "$current" | tr ':' ' ')
    for needed in $(needed_of "$current"); do
      case "$system" in *" $needed "*) continue ;; esac
      [ -n "''${resolved[$needed]:-}" ] && continue
      for dir in $rpath; do
        if [ -e "$dir/$needed" ]; then
          resolved[$needed]="$dir/$needed"
          queue+=( "$dir/$needed" )
          break
        fi
      done
      if [ -z "''${resolved[$needed]:-}" ]; then
        echo "isolated-bridge: $needed (needed by $current) is in no RUNPATH entry" >&2
        exit 1
      fi
    done
  done

  # The colliding sonames plus everything that names one of them.
  declare -A isolate=()
  for soname in ${lib.escapeShellArgs collidingSonames}; do
    if [ -z "''${resolved[$soname]:-}" ]; then
      echo "isolated-bridge: $soname is not a dependency of the bridge; drop it" >&2
      echo "  from collidingSonames or find out what replaced it." >&2
      exit 1
    fi
    isolate[$soname]=1
  done
  changed=1
  while [ "$changed" = 1 ]; do
    changed=0
    for soname in "''${!resolved[@]}"; do
      [ -n "''${isolate[$soname]:-}" ] && continue
      needs=" $(needed_of "''${resolved[$soname]}") "
      for isolated in "''${!isolate[@]}"; do
        case "$needs" in
          *" $isolated "*) isolate[$soname]=1; changed=1; break ;;
        esac
      done
    done
  done

  echo "isolating ''${#isolate[@]} of ''${#resolved[@]} libraries into ${imageDir}:"
  printf '  %s\n' "''${!isolate[@]}" | sort

  # The shim keeps its name; no distro ships that soname.
  install -m 0755 "${pyaccelstore.library}" "$dest/libpyaccelstore.so"
  for soname in "''${!isolate[@]}"; do
    install -m 0755 "''${resolved[$soname]}" "$dest/accelstore_$soname"
    patchelf --set-soname "accelstore_$soname" "$dest/accelstore_$soname"
  done

  for file in "$dest"/*; do
    for soname in "''${!isolate[@]}"; do
      case " $(needed_of "$file") " in
        *" $soname "*) patchelf --replace-needed "$soname" "accelstore_$soname" "$file" ;;
      esac
    done
    # This directory first, the recorded store paths behind it.
    patchelf --set-rpath "${imageDir}:$(patchelf --print-rpath "$file")" "$file"
  done
''
