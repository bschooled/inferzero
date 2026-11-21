#!/usr/bin/env bash
# Read-only power/clock survey while a GPU bench runs.
echo "== GPU busy =="
cat /sys/class/drm/card*/device/gpu_busy_percent 2>/dev/null
find /sys/class/drm -maxdepth 3 -name "gt*" -type d 2>/dev/null | head -5
for f in /sys/class/drm/card*/gt*/gt0_rps0_freq_status; do
  [ -e "$f" ] && cat "$f"
done
echo "== thermal =="
for z in /sys/class/thermal/thermal_zone*; do
  if [ -e "$z/temp" ]; then
    t=$(cat "$z/type")
    v=$(cat "$z/temp")
    printf "%s: %s\n" "$t" "$v"
  fi
done | head -12
echo "== platform profile =="
cat /sys/firmware/acpi/platform_profile 2>/dev/null
tr '\n' ' ' < /sys/firmware/acpi/platform_profile_choices 2>/dev/null
echo
echo "== cpu governor/current =="
for p in /sys/devices/system/cpu/cpufreq/policy*; do
  if [ -e "$p/scaling_governor" ]; then
    printf "%s gov=%s cur=%s max=%s\n" "$p" "$(cat $p/scaling_governor)" "$(cat $p/scaling_cur_freq)" "$(cat $p/scaling_max_freq)"
  fi
done
echo "== power daemon =="
powerprofilesctl get 2>/dev/null
systemctl is-enabled tlp 2>/dev/null
systemctl is-active power-profiles-daemon 2>/dev/null
echo "== RAPL limits uW =="
for c in /sys/class/powercap/intel-rapl:0/constraint_*_power_limit; do
  [ -e "$c" ] && printf "%s = %s\n" "$c" "$(cat $c)"
done
echo "== pkg power, 3 x 1s =="
for i in 1 2 3; do
  if [ -e /sys/class/powercap/intel-rapl:0/energy_uj ]; then
    e1=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)
    sleep 1
    e2=$(cat /sys/class/powercap/intel-rapl:0/energy_uj)
    printf "pkg %s mW\n" "$(( (e2-e1)/1000 ))"
  fi
done
echo "== intel_gpu_top =="
command -v intel_gpu_top >/dev/null && timeout 4 intel_gpu_top -s 2>/dev/null | tail -6 || echo "intel-gpu-tools not installed"
