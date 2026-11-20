# Which XMX tile each of a model's prompt GEMMs wants, measured on this GPU with the model itself (QUEUE: the A770's
# H8 GEMMs, G1). bench/xw_pergemm.sh is the same for Linux; the two must stay the same (change both together).
# For every tile setting it runs the real model's prompt (runz --bench-pp, the engine's own kernels on the model's own
# weights) with --prof, and records the GPU time of each GEMM kind (gemm_down, gemm_up, gemm_gate, gemm_qkv, gemm_o,
# gemm_ssm_out, ...): a model's GEMMs differ in shape (down: few columns, a long K; up / gate: many columns), and one
# tile for all of them can hide a tile that wins on some. Every setting gives the same results bit for bit (the engine
# checks the tile at start: "the H8 GEMMs on XMX with ... tiles"; a failed check keeps the default and is reported).
# Two rounds, the settings in the other order the second time; each GEMM's lower time is kept.
# Paste the whole output into the GitHub issue.
#   .\bench\xw_pergemm.ps1 -Model <Qwen3.8 27B or another Qwen dense gguf> [-Tokens 4096] [-Rounds 2] [-Quick]
# Settings: on 8-lane XMX (Arc A770 / A750 / A580, the Arc 140T): the default 32 x 16, 32 x 32, 16 x 32, 16 x 64,
# 64 x 16, and 1 / 4 column sub-groups per work-group; on 16-lane XMX (B-series): the default 32 x 64, 32 x 32,
# 32 x 96, 32 x 128, 16 x 128, 64 x 64, and 1 / 2 column sub-groups. -Quick: the first four of each.
# About 1-2 minutes per setting and round for the 27B at 4,096 tokens.
param(
  [Parameter(Mandatory = $true)][string]$Model,
  [int]$Tokens = 4096,
  [int]$Rounds = 2,
  [switch]$Quick
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$runz = Join-Path $root "bin\runz.exe"
if (-not (Test-Path $runz)) { throw "bin\runz.exe not found (run from the inferzero folder)" }
if (-not (Test-Path $Model)) { throw "model not found: $Model" }
$vars = @("ARC_GEMM_XW", "ARC_GEMM_XW_TM", "ARC_GEMM_XW_NC", "ARC_GEMM_XW_WGX", "ARC_GEMM_XW_GRF")
$inv = [Globalization.CultureInfo]::InvariantCulture

function Invoke-Runz([hashtable]$envs, [string[]]$a) {
  foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  $eap = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"  # (runz's stderr lines are output, not errors)
    foreach ($k in $envs.Keys) { Set-Item "Env:$k" "$($envs[$k])" }
    return (& $runz $Model @a 2>&1 | ForEach-Object { "$_" })
  } finally {
    $ErrorActionPreference = $eap
    foreach ($v in $vars) { Remove-Item "Env:$v" -ErrorAction SilentlyContinue }
  }
}

# the GPU's DPAS width, from a short run's log
$probe = Invoke-Runz @{} @("--bench-pp", "1024", "--n", "0")
$gpu = ($probe | Select-String -Pattern "^inferzero: GPU 0 \((.*)\):" | Select-Object -First 1)
# (16-lane GPUs say "no 8-lane sub-groups, so the 16-lane kernels"; Alchemist and the 140T have 8-lane sub-groups)
$lanes8 = -not [bool]($probe | Select-String -Pattern "no 8-lane sub-groups")
if (-not ($probe | Select-String -Pattern "XMX on for the prompt GEMMs|H8 GEMMs on XMX")) {
  Write-Output "note: this GPU's log shows no XMX prompt GEMMs; the tile settings below change nothing here"
}
$sets = [ordered]@{}
if ($lanes8) {
  $sets["32x16 (default)"] = @{}
  $sets["32x32"] = @{ ARC_GEMM_XW_NC = 2 }
  $sets["16x32"] = @{ ARC_GEMM_XW_TM = 16; ARC_GEMM_XW_NC = 2 }
  $sets["32x16 WGX 4"] = @{ ARC_GEMM_XW_WGX = 4 }
  if (-not $Quick) {
    $sets["16x64"] = @{ ARC_GEMM_XW_TM = 16; ARC_GEMM_XW_NC = 4 }
    $sets["64x16"] = @{ ARC_GEMM_XW_TM = 64 }
    $sets["32x16 WGX 1"] = @{ ARC_GEMM_XW_WGX = 1 }
    $sets["32x32 WGX 4"] = @{ ARC_GEMM_XW_NC = 2; ARC_GEMM_XW_WGX = 4 }
  }
} else {
  $sets["32x64 (default)"] = @{}
  $sets["32x32"] = @{ ARC_GEMM_XW_NC = 2 }
  $sets["32x96"] = @{ ARC_GEMM_XW_NC = 6 }
  $sets["32x128"] = @{ ARC_GEMM_XW_NC = 8 }
  if (-not $Quick) {
    $sets["16x128"] = @{ ARC_GEMM_XW_TM = 16; ARC_GEMM_XW_NC = 8 }
    $sets["64x64"] = @{ ARC_GEMM_XW_TM = 64 }
    $sets["32x64 WGX 1"] = @{ ARC_GEMM_XW_WGX = 1 }
    $sets["32x64 WGX 2"] = @{ ARC_GEMM_XW_WGX = 2 }
  }
}

Write-Output ("xw_pergemm: {0}, {1}, {2} tokens, {3} round(s), {4}" -f (Split-Path $Model -Leaf),
  $(if ($gpu) { $gpu.Matches[0].Groups[1].Value } else { "GPU ?" }), $Tokens, $Rounds,
  $(if ($lanes8) { "8-lane XMX (gemm_xw8)" } else { "16-lane XMX (gemm_xw)" }))
$ms = @{}      # set -> kernel -> lowest ms
$pp = @{}      # set -> best tok/s
$notes = @{}   # set -> the tile line the engine printed, or a failed check
$names = [System.Collections.Generic.List[string]]::new()
$keys = @($sets.Keys)
$rev = $keys.Clone(); [array]::Reverse($rev)
# (an untimed run at the same size first: the first measured setting was ~8 % slower than the rest without it, on a
#  GPU where every setting runs the same kernels)
$null = Invoke-Runz @{} @("--bench-pp", "$Tokens", "--n", "0")
for ($r = 1; $r -le $Rounds; $r++) {
  $order = if ($r % 2) { $keys } else { $rev }
  foreach ($s in $order) {
    Write-Host ("  round {0}: {1} ..." -f $r, $s)
    $o = Invoke-Runz $sets[$s] @("--bench-pp", "$Tokens", "--n", "0", "--prof")
    if (-not $ms.ContainsKey($s)) { $ms[$s] = @{} }
    foreach ($l in $o) {
      if ($l -match '^\s+(gemm_\w+)\s+([\d.]+) ms') {
        $k = $Matches[1]; $v = [double]::Parse($Matches[2], $inv)
        if (-not $names.Contains($k)) { $names.Add($k) }
        if (-not $ms[$s].ContainsKey($k) -or $v -lt $ms[$s][$k]) { $ms[$s][$k] = $v }
      } elseif ($l -match 'pp\d+ run \d+: [\d.]+ ms\s+->\s+([\d.]+) tokens/s') {
        $v = [double]::Parse($Matches[1], $inv)
        if (-not $pp.ContainsKey($s) -or $v -gt $pp[$s]) { $pp[$s] = $v }
      } elseif ($l -match 'H8 GEMMs on XMX with|gemm_xw.* differs|not supported') {
        $notes[$s] = $l.Trim()
      }
    }
  }
}

Write-Output ""
Write-Output "GPU time per GEMM kind (ms, the lower of the rounds); the last column is each GEMM's fastest setting"
$hdr = "{0,-16}" -f "GEMM"
foreach ($s in $keys) { $hdr += " {0,15}" -f $s }
Write-Output ($hdr + "   fastest")
$def = $keys[0]; $sumDef = 0.0; $sumBest = 0.0
foreach ($k in $names) {
  $line = "{0,-16}" -f $k; $best = $null; $bestv = [double]::MaxValue
  foreach ($s in $keys) {
    if ($ms[$s].ContainsKey($k)) {
      $v = $ms[$s][$k]; $line += " {0,15}" -f $v.ToString("F1", $inv)
      if ($v -lt $bestv) { $bestv = $v; $best = $s }
    } else { $line += " {0,15}" -f "-" }
  }
  if ($ms[$def].ContainsKey($k)) { $sumDef += $ms[$def][$k]; $sumBest += $bestv }
  $gain = if ($ms[$def].ContainsKey($k) -and $ms[$def][$k] -gt 0) { [string]::Format($inv, " ({0:+0.0;-0.0} %)", 100 * ($bestv / $ms[$def][$k] - 1)) } else { "" }
  Write-Output ($line + "   " + $best + $gain)
}
$line = "{0,-16}" -f "prompt tok/s"
foreach ($s in $keys) { $line += " {0,15}" -f $(if ($pp.ContainsKey($s)) { $pp[$s].ToString("F1", $inv) } else { "failed" }) }
Write-Output $line
Write-Output ""
Write-Output ([string]::Format($inv, "All GEMMs at the default: {0:F1} ms; each GEMM at its fastest setting: {1:F1} ms ({2:+0.0;-0.0} %).", $sumDef, $sumBest,
  $(if ($sumDef -gt 0) { 100 * ($sumBest / $sumDef - 1) } else { 0 })))
Write-Output "The tile each setting ran (from the log):"
foreach ($s in $keys) { Write-Output ("  {0,-16} {1}" -f $s, $(if ($notes.ContainsKey($s)) { $notes[$s] } else { "(no tile line)" })) }
